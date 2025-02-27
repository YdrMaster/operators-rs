use crate::{
    fuesd_softmax::AttnMask,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, LaunchError, MutPtr, TensorLayout,
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

    pub mask: AttnMask,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: usize,
    pub nkvh: usize,
    pub seq: usize,
    pub att: usize,
    pub dh: usize,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, LaunchError> {
        let Self {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            ..
        } = self;

        let &[nh_q, seq_q, dh_q] = &*q_layout.shape() else {
            return Err(rank_error("q", 3, q_layout.ndim()));
        };
        let &[nkvh_k, att_k, dh_k] = &*k_layout.shape() else {
            return Err(rank_error("k", 3, k_layout.ndim()));
        };
        let &[nkvh_v, att_v, dh_v] = &*v_layout.shape() else {
            return Err(rank_error("v", 3, v_layout.ndim()));
        };
        let &[nh_o, seq_o, dh_o] = &*o_layout.shape() else {
            return Err(rank_error("o", 3, o_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[q_layout.dt, k_layout.dt, v_layout.dt, o_layout.dt])?,
            nh: dim_distinct(&[nh_q, nh_o]).expect("nh mismatch"),
            nkvh: dim_distinct(&[nkvh_k, nkvh_v]).expect("nkvh mismatch"),
            seq: dim_distinct(&[seq_q, seq_o]).expect("seq mismatch"),
            att: dim_distinct(&[att_k, att_v]).expect("att mismatch"),
            dh: dim_distinct(&[dh_q, dh_k, dh_v, dh_o]).expect("dh mismatch"),
        })
    }
}
