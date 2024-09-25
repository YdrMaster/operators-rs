use crate::{
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub gate_layout: TensorLayout,
    pub gate_base: MutPtr<H>,
    pub up_layout: TensorLayout,
    pub up_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: MaybeDyn<usize>,
    pub d: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub fn new_layout(gate_layout: TensorLayout, up_layout: TensorLayout) -> Self {
        use std::ptr::{null, null_mut};
        Self {
            gate_layout,
            gate_base: null_mut(),
            up_layout,
            up_base: null(),
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            gate_layout,
            up_layout,
            ..
        } = self;

        let &[gn, gd] = gate_layout.shape() else {
            return Err(rank_error("gate", 2, gate_layout.ndim()));
        };
        let &[un, ud] = up_layout.shape() else {
            return Err(rank_error("up", 2, up_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[gate_layout.dt(), up_layout.dt()])?,
            n: dim_distinct(&[gn, un])?,
            d: dim_distinct(&[gd, ud])?,
        })
    }
}
