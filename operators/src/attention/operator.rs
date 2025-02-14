use super::{args::Meta, Args, Attention};
use crate::{
    dyn_, fuesd_softmax, get_static, mat_mul, rearrange, ByteOf, Hardware, LaunchError, QueueAlloc,
    SchemeError, TensorLayout, Workspace, WorkspaceCollector,
};
use ndarray_layout::ArrayLayout;
use std::marker::PhantomData;

pub struct Operator<Hardware, MatMul, Softmax, Rearrange> {
    mat_mul: MatMul,
    softmax: Softmax,
    rearrange: Rearrange,
    _phantom: PhantomData<Hardware>,
}

impl<H, M, S, R> Attention<H> for Operator<H, M, S, R>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    S: fuesd_softmax::FusedSoftmax<H>,
    R: rearrange::Rearrange<H>,
{
}

impl<H, M, S, R> crate::Operator for Operator<H, M, S, R>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    S: fuesd_softmax::FusedSoftmax<H>,
    R: rearrange::Rearrange<H>,
{
    type Hardware = H;
    type TopoNode = H;
    type Args = Args<H>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            mat_mul: M::new(node),
            softmax: S::new(node),
            rearrange: R::new(node),
            _phantom: PhantomData,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta {
            dt,
            nh,
            seq,
            att,
            dh,
            ..
        } = args.meta()?;
        let Args {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            ..
        } = args;

        // 如果不能保证 nh seq att dh 已知，用任意值初始化算子
        let (Some(&nh), Some(&seq), Some(&att), Some(&dh)) = (
            nh.get_static(),
            seq.get_static(),
            att.get_static(),
            dh.get_static(),
        ) else {
            let mut wc = WorkspaceCollector::new();

            let layout = TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]);
            wc.push_sub(self.mat_mul.scheme(
                &mat_mul::Args::new_null(layout.clone(), 1., layout.clone(), layout, 1.),
                max_workspace_size,
            )?);

            let layout = TensorLayout::new_dyn(dt, &[nh, seq, att], &[dyn_(); 3]);
            wc.push_sub(self.softmax.scheme(
                &fuesd_softmax::Args::new_null(args.mask, layout),
                max_workspace_size,
            )?);

            let layout = TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]);
            wc.push_sub(self.rearrange.scheme(
                &rearrange::Args::new_null(layout.clone(), layout),
                max_workspace_size,
            )?);

            return Ok(wc.cauculate(max_workspace_size));
        };

        let ele = dt.nbytes();
        let att_layout = TensorLayout::new_contiguous(dt, &[nh, seq, att]);
        let q_size = nh * seq * dh * ele;
        let att_size = nh * seq * att * ele;
        let workspace_size = max_workspace_size.saturating_sub(q_size + att_size);

        let mut wc = WorkspaceCollector::new();
        wc.push_base(q_size);
        wc.push_base(att_size);

        // att = q . k^T
        wc.push_sub(self.mat_mul.scheme(
            &mat_mul::Args::new_null(
                att_layout.clone(),
                0.,
                q_layout.clone(),
                k_layout.clone(),
                1.,
            ),
            workspace_size,
        )?);
        // att = softmax(att)
        wc.push_sub(self.softmax.scheme(
            &fuesd_softmax::Args::new_null(args.mask, att_layout.clone()),
            workspace_size,
        )?);
        // q = att . v
        wc.push_sub(self.mat_mul.scheme(
            &mat_mul::Args::new_null(q_layout.clone(), 0., att_layout, v_layout.clone(), 1.),
            workspace_size,
        )?);
        // o = rearrange(q)
        wc.push_sub(self.rearrange.scheme(
            &rearrange::Args::new_null(o_layout.clone(), q_layout.clone()),
            workspace_size,
        )?);

        Ok(wc.cauculate(max_workspace_size))
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta {
            dt,
            nh,
            nkvh,
            seq,
            att,
            dh,
        } = args.meta()?;
        let Args {
            mask,
            q_layout,
            q_base,
            k_layout,
            k_base,
            v_layout,
            v_base,
            o_layout,
            o_base,
        } = args;

        let &[nh_sq, seq_sq, dh_sq] = q_layout.strides() else {
            unreachable!()
        };
        let &[nkvh_sk, att_sk, dh_sk] = k_layout.strides() else {
            unreachable!()
        };

        let ele = dt.nbytes();
        get_static! {
            nh      seq    dh
            nh_sq   seq_sq dh_sq
            nkvh    att
            nkvh_sk att_sk dh_sk
        };

        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<3> {
            ArrayLayout::new(&shape, &strides, 0)
        }
        let qx = layout([nh, seq, dh], [nh_sq, seq_sq, dh_sq]).merge_be(0, 2);
        let k_layout = layout([nkvh, att, dh], [nkvh_sk, att_sk, dh_sk]).transpose(&[2, 1]);

        let q_size = if qx.is_none() { nh * seq * dh * ele } else { 0 };
        let att_size = nh * seq * att * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, q_size + att_size);
        let (q_buf, workspace) = workspace.split_at_mut(q_size);
        let (att_buf, workspace) = workspace.split_at_mut(att_size);

        let head_group = nh / nkvh;
        let (q_layout, qx_layout, q_base) = match qx {
            None => {
                let q_layout = TensorLayout::new_contiguous(dt, &[nh, seq, dh]);
                let qx_layout = TensorLayout::new_contiguous(dt, &[nkvh, head_group * seq, dh]);
                let q_base = q_buf.as_mut_ptr();
                self.rearrange.launch(
                    &rearrange::Args {
                        dst_layout: q_layout.clone(),
                        dst_base: q_base,
                        src_layout: args.q_layout.clone(),
                        src_base: args.q_base,
                    },
                    workspace,
                    queue_alloc,
                )?;
                (q_layout, qx_layout, q_base)
            }
            Some(qx) => {
                let qx = qx.tile_be(0, &[nkvh, head_group * seq]);
                let qx_layout = TensorLayout::new(dt, qx.shape(), qx.strides());
                (q_layout.clone(), qx_layout, *q_base)
            }
        };

        assert_eq!(k_layout.offset(), 0);
        let k_layout = TensorLayout::new(dt, k_layout.shape(), k_layout.strides());
        let att_mat_mul = TensorLayout::new_contiguous(dt, &[nkvh, head_group * seq, att]);
        let att_softmax = TensorLayout::new_contiguous(dt, &[nh, seq, att]);

        // att = q . k^T
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_mat_mul.clone(),
                c_base: att_buf.as_mut_ptr(),
                beta: 0.,
                a_layout: qx_layout.clone(),
                a_base: q_base,
                b_layout: k_layout,
                b_base: *k_base,
                alpha: (dh as f32).sqrt().recip(),
            },
            workspace,
            queue_alloc,
        )?;
        // att = softmax(att)
        self.softmax.launch(
            &fuesd_softmax::Args {
                att_mask: *mask,
                att_layout: att_softmax,
                att_base: att_buf.as_mut_ptr(),
            },
            workspace,
            queue_alloc,
        )?;
        // q = att . v
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: qx_layout.clone(),
                c_base: q_base,
                beta: 0.,
                a_layout: att_mat_mul,
                a_base: att_buf.as_ptr(),
                b_layout: v_layout.clone(),
                b_base: *v_base,
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )?;
        // o = rearrange(q)
        if q_base != *o_base {
            self.rearrange.launch(
                &rearrange::Args {
                    dst_layout: o_layout.clone(),
                    dst_base: *o_base,
                    src_layout: q_layout.clone(),
                    src_base: q_base,
                },
                workspace,
                queue_alloc,
            )?
        }
        Ok(())
    }
}
