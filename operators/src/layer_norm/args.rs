use crate::{
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, LaunchError, MutPtr, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,
    pub x_layout: TensorLayout,
    pub x_base: ConstPtr<H>,
    pub scale_layout: TensorLayout,
    pub scale_base: ConstPtr<H>,
    pub bias_layout: TensorLayout,
    pub bias_base: ConstPtr<H>,
    pub epsilon: f32,
}

pub(super) struct Meta {
    pub dt_a: DigitLayout,
    pub dt_w: DigitLayout,
    pub n: usize,
    pub d: usize,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, LaunchError> {
        let Self {
            y_layout: y,
            x_layout: x,
            scale_layout: scale,
            bias_layout: bias,
            ..
        } = self;

        let &[ny, dy] = &*y.shape() else {
            return Err(rank_error("y", 2, y.ndim()));
        };
        let &[nx, dx] = &*x.shape() else {
            return Err(rank_error("x", 2, x.ndim()));
        };
        let &[ds] = &*scale.shape() else {
            return Err(rank_error("scale", 1, scale.ndim()));
        };
        let &[db] = &*bias.shape() else {
            return Err(rank_error("bias", 1, bias.ndim()));
        };

        Ok(Meta {
            dt_a: type_distinct(&[y.dt, x.dt])?,
            dt_w: type_distinct(&[scale.dt, bias.dt])?,
            n: dim_distinct(&[ny, nx]).expect("m mismatch"),
            d: dim_distinct(&[dy, dx, ds, db]).expect("d mismatch"),
        })
    }
}
