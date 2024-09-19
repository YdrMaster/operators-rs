use super::{args::Meta, Args, AttnKVCached};
use crate::{attention, rearrange, utils::get_or_err};
use common::{
    algebraic, dyn_, locate_error, pass_match, Argument, ErrorPosition, Handle, QueueOf,
    TensorLayout, Workspace,
};
use digit_layout::DigitLayout;
use ndarray_layout::ArrayLayout;
use std::marker::PhantomData;

pub struct Operator<Handle, Rearrange, Attention> {
    dt: Option<DigitLayout>,
    nh: Argument<usize>,
    seq: Argument<usize>,
    dh: Argument<usize>,

    rearrange: Rearrange,
    attention: Attention,
    _phantom: PhantomData<Handle>,
}

impl<H, R, A> AttnKVCached<H> for Operator<H, R, A>
where
    H: Handle,
    R: rearrange::Rearrange<H>,
    A: attention::Attention<H>,
{
    fn workspace_size(&self) -> Option<usize> {
        let ele = self.dt?.nbytes()?;
        let nh = self.nh.get_static()?;
        let seq = self.seq.get_static()?;
        let dh = self.dh.get_static()?;
        let attn = self.attention.workspace_size()?;
        Some(nh * seq * dh * ele + attn)
    }
}

impl<H, R, A> common::Operator for Operator<H, R, A>
where
    H: Handle,
    R: rearrange::Rearrange<H>,
    A: attention::Attention<H>,
{
    type Handle = H;
    type Args = Args<H>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            dt: None,
            nh: dyn_(),
            seq: dyn_(),
            dh: dyn_(),

            rearrange: R::new(handle),
            attention: A::new(handle),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        use std::ptr::{null, null_mut};

        let Meta {
            dt,
            nh,
            nkvh,
            dh,
            seq,
        } = args.meta()?;

        self.dt = Some(dt);
        self.nh = nh;
        self.seq = seq;
        self.dh = dh;

        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]);
        self.rearrange.scheme(&rearrange::Args {
            dst_layout: layout.clone(),
            dst_base: null_mut(),
            src_layout: layout,
            src_base: null(),
        })?;

        self.attention.scheme(
            &attention::Meta {
                dt,
                nh,
                nkvh,
                seq,
                att: dyn_(),
                dh,
            }
            .into(),
        )
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta {
            dt,
            nh,
            nkvh,
            dh,
            seq,
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
            k_cache_layout,
            k_cache_base,
            v_cache_layout,
            v_cache_base,
            pos,
            workspace,
        } = args;

        pass_match! {
            &[_       , buf_k  , _     ] = k_cache_layout.shape();
            &[_       , buf_v  , _     ] = v_cache_layout.shape();
            &[nh_sq   , seq_sq , _     ] =       q_layout.strides();
            &[nkvh_skc, buf_skc, dh_skc] = k_cache_layout.strides();
            &[nkvh_svc, buf_svc, dh_svc] = k_cache_layout.strides();
            Some(attn_space) = self.attention.workspace_size();
        }
        get_or_err! {
            nh       seq     dh
            nh_sq    seq_sq
            nkvh
                     buf_k
                     buf_v
            nkvh_skc buf_skc dh_skc
            nkvh_svc buf_svc dh_svc
        };

        // 检查 cache 容量
        let att = pos + seq;
        if buf_k < att || buf_v < att {
            return Err(locate_error!("Out of cache buffer"));
        }
        // 如果 q 的前两维不连续则需要重整
        let rearrange_q = seq_sq * seq as isize != nh_sq;
        let ele = algebraic!(dt)?;
        if workspace.len < attn_space + if rearrange_q { nh * seq * dh * ele } else { 0 } {
            return Err(locate_error!("Out of workspace"));
        }
        let (q_layout, q_base) = if rearrange_q {
            let new = TensorLayout::new_contiguous(dt, &[nh, seq, dh]);
            let ptr = unsafe { workspace.ptr.byte_add(attn_space) };
            self.rearrange.launch(
                &rearrange::Args {
                    dst_layout: new.clone(),
                    dst_base: ptr,
                    src_layout: q_layout.clone(),
                    src_base: *q_base,
                },
                queue,
            )?;
            (new, ptr)
        } else {
            (q_layout.clone(), *q_base)
        };
        // 连接 kv cache
        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<3> {
            ArrayLayout::new(&shape, &strides, 0)
        }

        let kc_layout = layout([nkvh, buf_k, dh], [nkvh_skc, buf_skc, dh_skc]);
        let vc_layout = layout([nkvh, buf_v, dh], [nkvh_svc, buf_svc, dh_svc]);

        let k_cat = kc_layout.slice(1, *pos, 1, seq);
        let v_cat = vc_layout.slice(1, *pos, 1, seq);

        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, k_cat.shape(), k_cat.strides()),
                dst_base: unsafe { k_cache_base.byte_add(k_cat.offset()) },
                src_layout: k_layout.clone(),
                src_base: *k_base,
            },
            queue,
        )?;
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: TensorLayout::new(dt, v_cat.shape(), v_cat.strides()),
                dst_base: unsafe { v_cache_base.byte_add(k_cat.offset()) },
                src_layout: v_layout.clone(),
                src_base: *v_base,
            },
            queue,
        )?;
        // attention
        let k_layout = kc_layout.slice(1, 0, 1, att);
        let v_layout = vc_layout.slice(1, 0, 1, att);
        assert_eq!(k_layout.offset(), 0);
        assert_eq!(v_layout.offset(), 0);
        self.attention.launch(
            &attention::Args {
                q_layout,
                q_base,
                k_layout: TensorLayout::new(dt, k_layout.shape(), k_layout.strides()),
                k_base: *k_cache_base,
                v_layout: TensorLayout::new(dt, v_layout.shape(), v_layout.strides()),
                v_base: *v_cache_base,
                o_layout: o_layout.clone(),
                o_base: *o_base,
                workspace: Workspace {
                    ptr: workspace.ptr,
                    len: attn_space,
                },
            },
            queue,
        )
    }
}
