use crate::{
    dyn_,
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, ParamError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,

    pub x_layout: TensorLayout,
    pub x_base: ConstPtr<H>,

    pub w_gate_up_layout: TensorLayout,
    pub w_gate_up_base: ConstPtr<H>,

    pub w_down_layout: TensorLayout,
    pub w_down_base: ConstPtr<H>,
    pub down_alpha: f32,
    pub down_bias: bool,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nt: MaybeDyn<usize>,
    pub di: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(
        y_layout: TensorLayout,
        x_layout: TensorLayout,
        w_gate_up_layout: TensorLayout,
        w_down_layout: TensorLayout,
        down_alpha: f32,
        down_bias: bool,
    ) -> Self {
        use std::ptr::{null, null_mut};
        Self {
            y_layout,
            y_base: null_mut(),
            x_layout,
            x_base: null(),
            w_gate_up_layout,
            w_gate_up_base: null(),
            w_down_layout,
            w_down_base: null(),
            down_alpha,
            down_bias,
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let Self {
            y_layout,
            x_layout,
            w_gate_up_layout,
            w_down_layout,
            ..
        } = self;

        let &[nt_y, d_y] = y_layout.shape() else {
            return Err(rank_error("y", 2, y_layout.ndim()));
        };
        let &[nt_x, d_x] = x_layout.shape() else {
            return Err(rank_error("x", 2, x_layout.ndim()));
        };
        let &[d_gu, di_gu] = w_gate_up_layout.shape() else {
            return Err(rank_error("w_gate_up", 2, w_gate_up_layout.ndim()));
        };
        let &[di_d, d_d] = w_down_layout.shape() else {
            return Err(rank_error("w_down", 2, w_down_layout.ndim()));
        };

        let _ = dim_distinct(&[d_y, d_x, d_gu, d_d])?;

        Ok(Meta {
            dt: type_distinct(&[
                y_layout.dt(),
                x_layout.dt(),
                w_gate_up_layout.dt(),
                w_down_layout.dt(),
            ])?,
            nt: dim_distinct(&[nt_y, nt_x])?,
            di: match di_gu.get_static() {
                Some(&di2) => dim_distinct(&[di_d, (di2 / 2).into()])?,
                None => dyn_(),
            },
        })
    }
}
