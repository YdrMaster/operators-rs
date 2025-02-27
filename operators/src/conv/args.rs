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
    pub w_layout: TensorLayout,
    pub w_base: ConstPtr<H>,
    pub b_layout: TensorLayout,
    pub b_base: ConstPtr<H>,
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub pads: [usize; 4],
}

pub(crate) struct Meta {
    pub dt: DigitLayout,
    pub n: usize,
    pub m: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub hy: usize,
    pub wy: usize,
    pub hk: usize,
    pub wk: usize,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, LaunchError> {
        let Self {
            y_layout,
            x_layout,
            w_layout,
            b_layout,
            ..
        } = self;

        let &[ny, my, hy, wy] = &*y_layout.shape() else {
            return Err(rank_error("y", 4, y_layout.ndim()));
        };
        let &[n, c, h, w] = &*x_layout.shape() else {
            return Err(rank_error("x", 4, x_layout.ndim()));
        };
        let &[m, ck, hk, wk] = &*w_layout.shape() else {
            return Err(rank_error("w", 4, w_layout.ndim()));
        };
        let &[mb] = &*b_layout.shape() else {
            return Err(rank_error("b", 1, b_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[y_layout.dt, x_layout.dt, w_layout.dt, b_layout.dt])?,
            n: dim_distinct(&[n, ny]).expect("n mismatch"),
            m: dim_distinct(&[m, my, mb]).expect("m mismatch"),
            c: dim_distinct(&[c, ck]).expect("c mismatch"),
            h,
            w,
            hy,
            wy,
            hk,
            wk,
        })
    }
}
