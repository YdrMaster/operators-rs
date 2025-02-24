use crate::{
    fuesd_softmax::AttnMask,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    // q传入的是是吸收后的
    pub q_layout: TensorLayout,
    pub q_base: MutPtr<H>,

    pub kv_layout: TensorLayout,
    pub kv_base: ConstPtr<H>,

    pub absorb_layout: TensorLayout,
    pub absorb_base: ConstPtr<H>,

    pub qr_layout: TensorLayout,
    pub qr_base: ConstPtr<H>,

    pub kr_layout: TensorLayout,
    pub kr_base: ConstPtr<H>,

    pub o_layout: TensorLayout,
    pub o_base: MutPtr<H>,
    pub kv_cache_layout: TensorLayout,
    pub kv_cache_base: MutPtr<H>,

    pub kr_cache_layout: TensorLayout,
    pub kr_cache_base: MutPtr<H>,

    pub mask: AttnMask,
    pub pos: MaybeDyn<usize>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: MaybeDyn<usize>,
    pub seq: MaybeDyn<usize>,
    pub att: MaybeDyn<usize>,
    pub dkv: MaybeDyn<usize>,
    pub dv: MaybeDyn<usize>,
    pub dr: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            q_layout,
            kv_layout,
            absorb_layout,
            qr_layout,
            kr_layout,
            o_layout,
            kv_cache_layout,
            kr_cache_layout,
            ..
        } = self;

        let &[nh_q, seq_q, dkv_q] = q_layout.shape() else {
            return Err(rank_error("q", 3, q_layout.ndim()));
        };

        let &[nh_kv, attn_kv, dkv_kv] = kv_layout.shape() else {
            return Err(rank_error("kv", 3, kv_layout.ndim()));
        };
        let &[nh_a, dv_a, dkv_a] = absorb_layout.shape() else {
            return Err(rank_error("absorb", 3, absorb_layout.ndim()));
        };
        let &[nh_qr, seq_qr, dr_qr] = qr_layout.shape() else {
            return Err(rank_error("qr", 3, qr_layout.ndim()));
        };
        let &[nh_kr, att_kr, dr_kr] = kr_layout.shape() else {
            return Err(rank_error("kr", 3, kr_layout.ndim()));
        };
        let &[nh_o, seq_o, dv_o] = o_layout.shape() else {
            return Err(rank_error("o", 3, o_layout.ndim()));
        };
        let &[nh_kvc, _buf, dkv_kvc] = kv_cache_layout.shape() else {
            return Err(rank_error("k_cache", 3, kv_cache_layout.ndim()));
        };
        let &[nh_krc, _buf, dr_krc] = kr_cache_layout.shape() else {
            return Err(rank_error("v_cache", 3, kr_cache_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[
                q_layout.dt(),
                kv_layout.dt(),
                qr_layout.dt(),
                kr_layout.dt(),
                o_layout.dt(),
            ])?,
            nh: dim_distinct(&[nh_q, nh_kv, nh_a, nh_qr, nh_kr, nh_o, nh_krc, nh_kvc])?,
            seq: dim_distinct(&[seq_q, seq_o, seq_qr])?,
            att: dim_distinct(&[attn_kv, att_kr])?,
            dkv: dim_distinct(&[dkv_a, dkv_kv, dkv_q, dkv_kvc])?,
            dv: dim_distinct(&[dv_a, dv_o])?,
            dr: dim_distinct(&[dr_kr, dr_qr, dr_krc])?,
        })
    }
}
