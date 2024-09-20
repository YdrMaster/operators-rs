use crate::utils::{dim_distinct, rank_not_support, type_distinct, ConstPtr, MutPtr};
use common::{Argument, Handle, ParamError, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,
    pub x_layout: TensorLayout,
    pub x_base: ConstPtr<H>,
    pub w_layout: TensorLayout,
    pub w_base: ConstPtr<H>,
    pub epsilon: f32,
}

pub(super) struct Meta {
    pub dt_w: DigitLayout,
    pub dt_a: DigitLayout,
    pub n: Argument<usize>,
    pub d: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let Self {
            y_layout,
            x_layout,
            w_layout,
            ..
        } = self;

        let &[ny, dy] = y_layout.shape() else {
            return Err(rank_not_support("y", 2, y_layout.ndim()));
        };
        let &[nx, dx] = x_layout.shape() else {
            return Err(rank_not_support("x", 2, x_layout.ndim()));
        };
        let &[dw] = w_layout.shape() else {
            return Err(rank_not_support("w", 1, w_layout.ndim()));
        };

        Ok(Meta {
            dt_w: w_layout.dt(),
            dt_a: type_distinct(&[y_layout.dt(), x_layout.dt()])?,
            n: dim_distinct(&[ny, nx])?,
            d: dim_distinct(&[dy, dx, dw])?,
        })
    }
}
