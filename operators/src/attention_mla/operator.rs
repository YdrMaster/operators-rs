use super::{args::Meta, Args, AttentionMLA};
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

impl<H, M, S, R> AttentionMLA<H> for Operator<H, M, S, R>
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
        // TODO
        Ok(0)
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
            seq,
            att,
            dkv,
            dv,
            dr,
        } = args.meta()?;
        let Args {
            q_layout,
            q_base,
            kv_layout,
            kv_base,
            absorb_layout,
            absorb_base,
            qr_layout,
            qr_base,
            kr_layout,
            kr_base,
            o_layout,
            o_base,
            mask,
        } = args;

        let &[nh_skv, att_skv, dkv_skv] = kv_layout.strides() else {
            unreachable!()
        };
        let &[nh_skr, att_skr, dr_skr] = kr_layout.strides() else {
            unreachable!()
        };
        let &[nh_sa, dv_sa, dkv_sa] = absorb_layout.strides() else {
            unreachable!()
        };
        let &[nh_so, seq_so, dv_so] = o_layout.strides() else {
            unreachable!()
        };
        let ele = dt.nbytes();
        get_static! {
            nh      seq     dkv     dr
            nh_skv  att_skv  dkv_skv
            nh_skr  att_skr  dr_skr
            nh_sa   dv_sa    dkv_sa
            nh_so   seq_so   dv_so
            dv      att
        };

        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<3> {
            ArrayLayout::new(&shape, &strides, 0)
        }
        let kv_first_layout = layout([nh, att, dkv], [nh_skv, att_skv, dkv_skv]).transpose(&[2, 1]);
        let kr_layout = layout([nh, att, dr], [nh_skr, att_skr, dr_skr]).transpose(&[2, 1]);
        let a_layout = layout([nh, dv, dkv], [nh_sa, dv_sa, dkv_sa]).transpose(&[2, 1]);
        let att_w_layout = TensorLayout::new_contiguous(dt, &[nh, seq, att]);
        let attn_t_layout = TensorLayout::new_contiguous(dt, &[nh, seq, dkv]);
        let att_w_size = nh * seq * att * ele;
        let att_t_size = nh * seq * dkv * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, att_w_size + att_t_size);
        let (att_w_buf, workspace) = workspace.split_at_mut(att_w_size);
        let (attn_t_buf, workspace) = workspace.split_at_mut(att_t_size);

        let kv_first_layout =
            TensorLayout::new(dt, kv_first_layout.shape(), kv_first_layout.strides());
        let kr_layout = TensorLayout::new(dt, kr_layout.shape(), kr_layout.strides());
        let a_layout = TensorLayout::new(dt, a_layout.shape(), a_layout.strides());
        // att_w = qr*kr^T + q*kv^T
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_w_layout.clone(),
                c_base: att_w_buf.as_mut_ptr(),
                beta: 0.,
                a_layout: qr_layout.clone(),
                a_base: *qr_base,
                b_layout: kr_layout.clone(),
                b_base: *kr_base,
                alpha: ((dv + dr) as f32).sqrt().recip(),
            },
            workspace,
            queue_alloc,
        )?;
       
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_w_layout.clone(),
                c_base: att_w_buf.as_mut_ptr(),
                beta: 1.,
                a_layout: q_layout.clone(),
                a_base: *q_base,
                b_layout: kv_first_layout.clone(),
                b_base: *kv_base,
                alpha: ((dv + dr) as f32).sqrt().recip(),
            },
            workspace,
            queue_alloc,
        )?;
        // att_w = softmax(att)
        self.softmax.launch(
            &fuesd_softmax::Args {
                att_mask: *mask,
                att_layout: att_w_layout.clone(),
                att_base: att_w_buf.as_mut_ptr(),
            },
            workspace,
            queue_alloc,
        )?;
        // attn_t=att_o*kv
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: attn_t_layout.clone(),
                c_base: attn_t_buf.as_mut_ptr(),
                beta: 0.,
                a_layout: att_w_layout.clone(),
                a_base: att_w_buf.as_ptr(),
                b_layout: kv_layout.clone(),
                b_base: *kv_base,
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )?;

        // attn =attn_t*absorb^T
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: o_layout.clone(),
                c_base: *o_base,
                beta: 0.,
                a_layout: attn_t_layout.clone(),
                a_base: attn_t_buf.as_ptr(),
                b_layout: a_layout.clone(),
                b_base: *absorb_base,
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )?;

        Ok(())
    }
}
