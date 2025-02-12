use crate::{
    dyn_,
    fuesd_softmax::AttnMask,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;
use std::ptr::{null, null_mut};

pub struct Args<H: Hardware> {
    pub q_layout: TensorLayout,
    pub q_base: MutPtr<H>,

    pub k_layout: TensorLayout,
    pub k_base: ConstPtr<H>,

    pub v_layout: TensorLayout,
    pub v_base: ConstPtr<H>,

    pub o_layout: TensorLayout,
    pub o_base: MutPtr<H>,

    pub mask: AttnMask,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: MaybeDyn<usize>,
    pub nkvh: MaybeDyn<usize>,
    pub seq: MaybeDyn<usize>,
    pub att: MaybeDyn<usize>,
    pub dh: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub(crate) fn new_null(
        mask: AttnMask,
        dt: DigitLayout,
        nh: MaybeDyn<usize>,
        nkvh: MaybeDyn<usize>,
        seq: MaybeDyn<usize>,
        att: MaybeDyn<usize>,
        dh: MaybeDyn<usize>,
    ) -> Self {
        let qo_layout = TensorLayout::new_dyn(dt, &[nh, seq, dh], &[dyn_(); 3]);
        let kv_layout = TensorLayout::new_dyn(dt, &[nkvh, att, dh], &[dyn_(); 3]);
        Self {
            q_layout: qo_layout.clone(),
            q_base: null_mut(),
            k_layout: kv_layout.clone(),
            k_base: null(),
            v_layout: kv_layout,
            v_base: null(),
            o_layout: qo_layout,
            o_base: null_mut(),
            mask,
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            ..
        } = self;

        let &[nh_q, seq_q, dh_q] = q_layout.shape() else {
            return Err(rank_error("q", 3, q_layout.ndim()));
        };
        let &[nkvh_k, att_k, dh_k] = k_layout.shape() else {
            return Err(rank_error("k", 3, k_layout.ndim()));
        };
        let &[nkvh_v, att_v, dh_v] = v_layout.shape() else {
            return Err(rank_error("v", 3, v_layout.ndim()));
        };
        let &[nh_o, seq_o, dh_o] = o_layout.shape() else {
            return Err(rank_error("o", 3, o_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[q_layout.dt(), k_layout.dt(), v_layout.dt(), o_layout.dt()])?,
            nh: dim_distinct(&[nh_q, nh_o])?,
            nkvh: dim_distinct(&[nkvh_k, nkvh_v])?,
            seq: dim_distinct(&[seq_q, seq_o])?,
            att: dim_distinct(&[att_k, att_v])?,
            dh: dim_distinct(&[dh_q, dh_k, dh_v, dh_o])?,
        })
    }
}
