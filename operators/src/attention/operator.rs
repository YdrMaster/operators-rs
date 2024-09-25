use super::{args::Meta, Args, Attention};
use crate::{
    dyn_, fuesd_softmax, get_static, mat_mul, rearrange, utils::sizeof, ByteOf, Hardware,
    LaunchError, QueueAlloc, SchemeError, TensorLayout, Workspace, WorkspaceCollector,
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
    type Args = Args<H>;

    fn new(processor: &Self::Hardware) -> Self {
        Self {
            mat_mul: M::new(processor),
            softmax: S::new(processor),
            rearrange: R::new(processor),
            _phantom: PhantomData,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta {
            dt, nh, seq, att, ..
        } = args.meta()?;
        let Args {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            ..
        } = args;

        // 如果不能保证 nh seq att 已知，用任意值初始化算子
        let (Some(&nh), Some(&seq), Some(&att)) =
            (nh.get_static(), seq.get_static(), att.get_static())
        else {
            let mut wc = WorkspaceCollector::new();

            let layout = TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]);
            wc.push_sub(self.mat_mul.scheme(
                &mat_mul::Args::new_null(layout.clone(), 1., layout.clone(), layout, 1.),
                max_workspace_size,
            )?);

            let layout = TensorLayout::new_dyn(dt, &[nh, seq, att], &[dyn_(); 3]);
            wc.push_sub(
                self.softmax
                    .scheme(&fuesd_softmax::Args::new_null(layout), max_workspace_size)?,
            );

            let layout = TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]);
            wc.push_sub(self.rearrange.scheme(
                &rearrange::Args::new_null(layout.clone(), layout),
                max_workspace_size,
            )?);

            return Ok(wc.cauculate(max_workspace_size));
        };

        let ele = sizeof(dt)?;
        let att_layout = TensorLayout::new_contiguous(dt, &[nh, seq, att]);
        let att_size = nh * seq * att * ele;
        let workspace_size = max_workspace_size.saturating_sub(att_size);

        let mut wc = WorkspaceCollector::new();
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
            &fuesd_softmax::Args::new_null(att_layout.clone()),
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

        let ele = sizeof(dt)?;
        get_static! {
            nh      seq    dh
            nh_sq   seq_sq dh_sq
            nkvh    att
            nkvh_sk att_sk dh_sk
        };

        let att_size = nh * seq * att * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, att_size);
        let (att_buf, workspace) = workspace.split_at_mut(att_size);

        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<4> {
            ArrayLayout::new(&shape, &strides, 0)
        }
        let att_layout = layout([nh, seq, dh], [nh_sq, seq_sq, dh_sq]);
        let k_layout = layout([nkvh, att, dh], [nkvh_sk, att_sk, dh_sk]);

        let head_group = nh / nkvh;
        let att_layout = att_layout.tile_be(0, &[nkvh, head_group]).merge(1..3);
        let k_layout = k_layout.transpose(&[2, 1]);

        assert_eq!(att_layout.offset(), 0);
        assert_eq!(k_layout.offset(), 0);
        let att_layout = TensorLayout::new(dt, att_layout.shape(), att_layout.strides());
        let k_layout = TensorLayout::new(dt, k_layout.shape(), k_layout.strides());
        let att_mat_mul = TensorLayout::new_contiguous(dt, &[nkvh, head_group * seq, att]);
        let att_softmax = TensorLayout::new_contiguous(dt, &[nh, seq, att]);

        // att = q . k^T
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_mat_mul.clone(),
                c_base: att_buf.as_mut_ptr(),
                beta: 0.,
                a_layout: att_layout.clone(),
                a_base: *q_base,
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
                att_layout: att_softmax,
                att_base: att_buf.as_mut_ptr(),
            },
            workspace,
            queue_alloc,
        )?;
        // q = att . v
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_layout.clone(),
                c_base: *q_base,
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
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: o_layout.clone(),
                dst_base: *o_base,
                src_layout: q_layout.clone(),
                src_base: *q_base,
            },
            workspace,
            queue_alloc,
        )
    }
}
