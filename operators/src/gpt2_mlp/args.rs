use crate::{
    utils::{dim_distinct, rank_error, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,

    pub up_weight_layout: TensorLayout,
    pub up_weight_base: ConstPtr<H>,

    pub up_bias_layout: TensorLayout,
    pub up_bias_base: ConstPtr<H>,

    pub down_weight_layout: TensorLayout,
    pub down_weight_base: ConstPtr<H>,

    pub down_bias_layout: TensorLayout,
    pub down_bias_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nt: MaybeDyn<usize>,
    pub di: MaybeDyn<usize>,
    pub d: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(
        y_layout: TensorLayout,
        up_weight_layout: TensorLayout,
        up_bias_layout: TensorLayout,
        down_weight_layout: TensorLayout,
        down_bias_layout: TensorLayout,
    ) -> Self {
        use std::ptr::{null, null_mut};
        Self {
            y_layout,
            y_base: null_mut(),
            up_weight_layout,
            up_weight_base: null(),
            up_bias_layout,
            up_bias_base: null(),
            down_weight_layout,
            down_weight_base: null(),
            down_bias_layout,
            down_bias_base: null(),
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            y_layout,
            up_weight_layout,
            up_bias_layout,
            down_weight_layout,
            down_bias_layout,
            ..
        } = self;
        // ffn_up_weight [d,di]
        // ffn_up_bias [di]
        // ffn_down_weight [di,d]
        //  ffn_down_bias [d]
        let &[nt_y, d_y] = y_layout.shape() else {
            return Err(rank_error("y", 2, y_layout.ndim()));
        };
        let &[d_uw, di_uw] = up_weight_layout.shape() else {
            return Err(rank_error("up_weight_layout", 2, up_weight_layout.ndim()));
        };
        let &[di_ub] = up_bias_layout.shape() else {
            return Err(rank_error("up_bias_layout", 1, up_bias_layout.ndim()));
        };
        let &[di_dw, d_dw] = down_weight_layout.shape() else {
            return Err(rank_error("down_weight_layout", 2, up_weight_layout.ndim()));
        };
        let &[d_db] = down_bias_layout.shape() else {
            return Err(rank_error("down_bias_layout", 1, up_bias_layout.ndim()));
        };
        let _ = dim_distinct(&[d_y, d_uw, d_dw, d_db])?;
        let _ = dim_distinct(&[di_uw, di_ub, di_dw])?;
        Ok(Meta {
            dt: type_distinct(&[
                y_layout.dt(),
                up_bias_layout.dt(),
                up_bias_layout.dt(),
                down_weight_layout.dt(),
                down_bias_layout.dt(),
            ])?,
            nt: nt_y,
            di: dim_distinct(&[di_uw, di_ub, di_dw])?,
            d: d_y,
        })
    }
}
