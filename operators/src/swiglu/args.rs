use crate::utils::{dim_distinct, rank_not_support, type_distinct, ConstPtr, MutPtr};
use common::{Argument, Handle, ParamError, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub gate_layout: TensorLayout,
    pub gate_base: MutPtr<H>,
    pub up_layout: TensorLayout,
    pub up_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: Argument<usize>,
    pub d: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let Self {
            gate_layout,
            up_layout,
            ..
        } = self;

        let &[gn, gd] = gate_layout.shape() else {
            return Err(rank_not_support("gate", 2, gate_layout.ndim()));
        };
        let &[un, ud] = up_layout.shape() else {
            return Err(rank_not_support("up", 2, up_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[gate_layout.dt(), up_layout.dt()])?,
            n: dim_distinct(&[gn, un])?,
            d: dim_distinct(&[gd, ud])?,
        })
    }
}
