use super::{args::Meta, Args, AttnKVCached};
use crate::{
    attention, rearrange, shape_mismatch, ByteOf, Hardware, LaunchError, QueueAlloc, TensorLayout,
};
use ndarray_layout::ArrayLayout;
use std::marker::PhantomData;

pub struct Operator<Hardware, Rearrange, Attention> {
    rearrange: Rearrange,
    attention: Attention,
    _phantom: PhantomData<Hardware>,
}

impl<H, R, A> AttnKVCached<H> for Operator<H, R, A>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    A: attention::Attention<H>,
{
}

impl<H, R, A> crate::Operator for Operator<H, R, A>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    A: attention::Attention<H>,
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
            dt, nkvh, dh, seq, ..
        } = args.meta()?;
        let &Args {
            ref q_layout,
            q_base,
            ref k_layout,
            k_base,
            ref v_layout,
            v_base,
            ref o_layout,
            o_base,
            ref k_cache_layout,
            k_cache_base,
            ref v_cache_layout,
            v_cache_base,
            mask,
            pos,
        } = args;

        let &[_, buf_k, _] = &*k_cache_layout.shape() else {
            unreachable!()
        };
        let &[_, buf_v, _] = &*v_cache_layout.shape() else {
            unreachable!()
        };
        let &[nkvh_skc, buf_skc, dh_skc] = k_cache_layout.strides() else {
            unreachable!()
        };
        let &[nkvh_svc, buf_svc, dh_svc] = k_cache_layout.strides() else {
            unreachable!()
        };

        // 检查 cache 容量
        let att = pos + seq;
        if buf_k < att || buf_v < att {
            return Err(shape_mismatch("Out of cache buffer"));
        }
        // 连接 kv cache
        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<3> {
            ArrayLayout::new(&shape, &strides, 0)
        }

        let kc_layout = layout([nkvh, buf_k, dh], [nkvh_skc, buf_skc, dh_skc]);
        let vc_layout = layout([nkvh, buf_v, dh], [nkvh_svc, buf_svc, dh_svc]);

        let k_cat = kc_layout.slice(1, pos, 1, seq);
        let v_cat = vc_layout.slice(1, pos, 1, seq);

        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, k_cat.shape(), k_cat.strides()),
                dst_base: unsafe { k_cache_base.byte_add(k_cat.offset() as _) },
                src_layout: k_layout.clone(),
                src_base: k_base,
            },
            workspace,
            queue_alloc,
        )?;
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, v_cat.shape(), v_cat.strides()),
                dst_base: unsafe { v_cache_base.byte_add(k_cat.offset() as _) },
                src_layout: v_layout.clone(),
                src_base: v_base,
            },
            workspace,
            queue_alloc,
        )?;
        // attention
        let k_layout = kc_layout.slice(1, 0, 1, att);
        let v_layout = vc_layout.slice(1, 0, 1, att);
        assert_eq!(k_layout.offset(), 0);
        assert_eq!(v_layout.offset(), 0);
        self.attention.launch(
            &attention::Args {
                mask,
                q_layout: q_layout.clone(),
                q_base,
                k_layout: TensorLayout::new(dt, k_layout.shape(), k_layout.strides()),
                k_base: k_cache_base,
                v_layout: TensorLayout::new(dt, v_layout.shape(), v_layout.strides()),
                v_base: v_cache_base,
                o_layout: o_layout.clone(),
                o_base,
            },
            workspace,
            queue_alloc,
        )
    }
}
