use crate::utils::{dim_distinct, rank_not_support, type_distinct, ConstPtr, MutPtr};
use common::{dyn_, Argument, Handle, ParamError, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
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

    pub workspace_size: usize,
    pub workspace: MutPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nt: Argument<usize>,
    pub di: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let Self {
            y_layout,
            x_layout,
            w_gate_up_layout,
            w_down_layout,
            ..
        } = self;

        let &[nt_y, d_y] = y_layout.shape() else {
            return Err(rank_not_support("y", 2, y_layout.ndim()));
        };
        let &[nt_x, d_x] = x_layout.shape() else {
            return Err(rank_not_support("x", 2, x_layout.ndim()));
        };
        let &[d_gu, di_gu] = w_gate_up_layout.shape() else {
            return Err(rank_not_support("w_gate_up", 2, w_gate_up_layout.ndim()));
        };
        let &[di_d, d_d] = w_down_layout.shape() else {
            return Err(rank_not_support("w_down", 2, w_down_layout.ndim()));
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
