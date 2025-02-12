use crate::{
    fuesd_softmax::AttnMask,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub q_layout: TensorLayout,
    pub q_base: MutPtr<H>,

    pub k_layout: TensorLayout,
    pub k_base: ConstPtr<H>,

    pub v_layout: TensorLayout,
    pub v_base: ConstPtr<H>,

    pub o_layout: TensorLayout,
    pub o_base: MutPtr<H>,

    pub k_cache_layout: TensorLayout,
    pub k_cache_base: MutPtr<H>,

    pub v_cache_layout: TensorLayout,
    pub v_cache_base: MutPtr<H>,

    pub mask: AttnMask,
    pub pos: MaybeDyn<usize>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: MaybeDyn<usize>,
    pub nkvh: MaybeDyn<usize>,
    pub dh: MaybeDyn<usize>,

    pub seq: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_null(
        q_layout: TensorLayout,
        k_layout: TensorLayout,
        v_layout: TensorLayout,
        o_layout: TensorLayout,
        k_cache_layout: TensorLayout,
        v_cache_layout: TensorLayout,
        mask: AttnMask,
        pos: MaybeDyn<usize>,
    ) -> Self {
        use std::ptr::{null, null_mut};
        Self {
            q_layout,
            q_base: null_mut(),
            k_layout,
            k_base: null(),
            v_layout,
            v_base: null(),
            o_layout,
            o_base: null_mut(),
            k_cache_layout,
            k_cache_base: null_mut(),
            v_cache_layout,
            v_cache_base: null_mut(),
            mask,
            pos,
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            k_cache_layout,
            v_cache_layout,
            ..
        } = self;

        let &[nh_q, seq_q, dh_q] = q_layout.shape() else {
            return Err(rank_error("q", 3, q_layout.ndim()));
        };
        let &[nkvh_k, seq_k, dh_k] = k_layout.shape() else {
            return Err(rank_error("k", 3, k_layout.ndim()));
        };
        let &[nkvh_v, seq_v, dh_v] = v_layout.shape() else {
            return Err(rank_error("v", 3, v_layout.ndim()));
        };
        let &[nh_o, seq_o, dh_o] = o_layout.shape() else {
            return Err(rank_error("o", 3, o_layout.ndim()));
        };
        let &[nkvh_kc, _buf, dh_kc] = k_cache_layout.shape() else {
            return Err(rank_error("k_cache", 3, k_cache_layout.ndim()));
        };
        let &[nkvh_vc, _buf, dh_vc] = v_cache_layout.shape() else {
            return Err(rank_error("v_cache", 3, v_cache_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[
                q_layout.dt(),
                k_layout.dt(),
                v_layout.dt(),
                o_layout.dt(),
                k_cache_layout.dt(),
                v_cache_layout.dt(),
            ])?,
            nh: dim_distinct(&[nh_q, nh_o])?,
            nkvh: dim_distinct(&[nkvh_k, nkvh_v, nkvh_kc, nkvh_vc])?,
            dh: dim_distinct(&[dh_q, dh_k, dh_v, dh_o, dh_kc, dh_vc])?,
            seq: dim_distinct(&[seq_q, seq_k, seq_v, seq_o])?,
        })
    }
}
