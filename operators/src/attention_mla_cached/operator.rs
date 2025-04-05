use super::{args::Meta, Args, AttMLACached};
use crate::{
    attention_mla, get_static, rearrange, shape_mismatch, ByteOf, Hardware, LaunchError,
    QueueAlloc, SchemeError, TensorLayout,
};
use ndarray_layout::ArrayLayout;
use std::marker::PhantomData;
pub struct Operator<Hardware, Rearrange, Attention> {
    rearrange: Rearrange,
    attention: Attention,
    _phantom: PhantomData<Hardware>,
}

impl<H, R, A> AttMLACached<H> for Operator<H, R, A>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    A: attention_mla::AttentionMLA<H>,
{
}

impl<H, R, A> crate::Operator for Operator<H, R, A>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    A: attention_mla::AttentionMLA<H>,
{
    type Hardware = H;
    type TopoNode = H;
    type Args = Args<H>;
    fn new(node: &Self::TopoNode) -> Self {
        Self {
            rearrange: R::new(node),
            attention: A::new(node),
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
            kv_cache_layout,
            kv_cache_base,
            kr_cache_layout,
            kr_cache_base,
            mask,
            pos,
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

        let &[_, buf_kv, _] = kv_cache_layout.shape() else {
            unreachable!()
        };
        let &[_, buf_kr, _] = kr_cache_layout.shape() else {
            unreachable!()
        };
        let &[nh_skvc, buf_skvc, dh_skvc] = kv_cache_layout.strides() else {
            unreachable!()
        };
        let &[nh_skrc, buf_skrc, dh_skrc] = kr_cache_layout.strides() else {
            unreachable!()
        };
        let ele = dt.nbytes();
        get_static! {
            nh      seq     dkv     dr
            pos
            buf_kv  buf_kr
            nh_skvc buf_skvc dh_skvc
            nh_skrc buf_skrc dh_skrc

        };

        // 检查 cache 容量
        let att = pos + seq;
        if buf_kr < att || buf_kv < att {
            return Err(shape_mismatch("Out of cache buffer").into());
        }
        // 连接 kv cache
        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<3> {
            ArrayLayout::new(&shape, &strides, 0)
        }

        let kvc_layout = layout([nh, buf_kv, dkv], [nh_skvc, buf_skvc, dh_skvc]);
        let krc_layout = layout([nh, buf_kr, dr], [nh_skrc, buf_skrc, dh_skrc]);

        let kv_cat = kvc_layout.slice(1, pos, 1, seq);
        let kr_cat = krc_layout.slice(1, pos, 1, seq);

        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, kv_cat.shape(), kv_cat.strides()),
                dst_base: unsafe { kv_cache_base.byte_add(kv_cat.offset() as _) },
                src_layout: kv_layout.clone(),
                src_base: *kv_base,
            },
            workspace,
            queue_alloc,
        )?;
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, kr_cat.shape(), kr_cat.strides()),
                dst_base: unsafe { kr_cache_base.byte_add(kr_cat.offset() as _) },
                src_layout: kr_layout.clone(),
                src_base: *kr_base,
            },
            workspace,
            queue_alloc,
        )?;
        // attention
        let kv_layout = kvc_layout.slice(1, 0, 1, att);
        let kr_layout = krc_layout.slice(1, 0, 1, att);
        assert_eq!(kv_layout.offset(), 0);
        assert_eq!(kr_layout.offset(), 0);
        self.attention.launch(
            &attention_mla::Args {
                mask: *mask,
                q_layout: q_layout.clone(),
                q_base: *q_base,
                kv_layout: TensorLayout::new(dt, kv_layout.shape(), kv_layout.strides()),
                kv_base: *kv_cache_base,
                kr_layout: TensorLayout::new(dt, kr_layout.shape(), kr_layout.strides()),
                kr_base: *kr_cache_base,
                absorb_layout: absorb_layout.clone(),
                absorb_base: *absorb_base,
                qr_layout: qr_layout.clone(),
                qr_base: *qr_base,
                o_layout: o_layout.clone(),
                o_base: *o_base,
            },
            workspace,
            queue_alloc,
        )?;
        Ok(())
    }
}
