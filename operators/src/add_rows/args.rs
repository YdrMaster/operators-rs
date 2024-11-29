use crate::{
    type_not_support,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::{DigitLayout, LayoutContent::Unsigned};
use std::ptr::{null, null_mut};

#[derive(Clone)]
pub struct Args<H: Hardware> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
    pub idx_layout: TensorLayout,
    pub idx_base: ConstPtr<H>,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(
        dst_layout: TensorLayout,
        src_layout: TensorLayout,
        idx_layout: TensorLayout,
    ) -> Self {
        Self {
            dst_layout,
            dst_base: null_mut(),
            src_layout,
            src_base: null(),
            idx_layout,
            idx_base: null(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Meta {
    pub dt: DigitLayout,
    pub dt_idx: DigitLayout,
    pub batch: MaybeDyn<usize>,
    pub m: MaybeDyn<usize>,
    pub n: MaybeDyn<usize>,
    pub k: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            dst_layout: dst,
            src_layout: src,
            idx_layout: idx,
            ..
        } = self;

        let dt = type_distinct(&[dst.dt(), src.dt()])?;
        let dt_idx = idx.dt();
        if !matches!(dt_idx.decode(), Unsigned { .. }) {
            return Err(type_not_support(format!(
                "data type {dt_idx} is not supported, must be unsigned integers"
            )));
        }

        let &[batch, m, n] = dst.shape() else {
            return Err(rank_error("dst", 3, dst.ndim()));
        };
        let &[k, n_] = src.shape() else {
            return Err(rank_error("src", 2, src.ndim()));
        };
        let &[batch_, m_] = idx.shape() else {
            return Err(rank_error("idx", 2, idx.ndim()));
        };

        Ok(Meta {
            dt,
            dt_idx,
            batch: dim_distinct(&[batch, batch_])?,
            m: dim_distinct(&[m, m_])?,
            n: dim_distinct(&[n, n_])?,
            k,
        })
    }
}
