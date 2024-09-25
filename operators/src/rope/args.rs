use crate::{
    type_not_support,
    utils::{dim_distinct, rank_error},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub t_layout: TensorLayout,
    pub t_base: MutPtr<H>,
    pub p_layout: TensorLayout,
    pub p_base: ConstPtr<H>,
    pub sin_layout: TensorLayout,
    pub sin_base: ConstPtr<H>,
    pub cos_layout: TensorLayout,
    pub cos_base: ConstPtr<H>,
    pub theta: f32,
}

pub(super) struct Meta {
    pub dt_t: DigitLayout,
    pub dt_p: DigitLayout,
    pub nt: MaybeDyn<usize>,
    #[allow(dead_code)]
    pub dh: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            t_layout,
            p_layout,
            sin_layout,
            cos_layout,
            ..
        } = self;

        let &[nt, _, dh] = t_layout.shape() else {
            return Err(rank_error("t", 3, t_layout.ndim()));
        };
        let &[np] = p_layout.shape() else {
            return Err(rank_error("p", 1, p_layout.ndim()));
        };
        let &[_, dh_sin] = sin_layout.shape() else {
            return Err(rank_error("sin", 2, sin_layout.ndim()));
        };
        let &[_, dh_cos] = cos_layout.shape() else {
            return Err(rank_error("cos", 2, cos_layout.ndim()));
        };

        let dt_t = t_layout.dt();
        let dt_p = p_layout.dt();
        use digit_layout::LayoutContent::{Real, Unsigned};
        // tokens must be floating-point numbers
        if !matches!(dt_t.decode(), Real { exponent: 1.., .. },) {
            return Err(type_not_support(format!(
                "data type {dt_t} is not supported, must be floating-point numbers",
            )));
        }
        // positions must be unsigned integers
        if !matches!(dt_p.decode(), Unsigned { .. }) {
            return Err(type_not_support(format!(
                "data type {dt_p} is not supported, must be unsigned integers"
            )));
        }
        Ok(Meta {
            dt_t,
            dt_p,
            nt: dim_distinct(&[nt, np])?,
            dh: dim_distinct(&[dh, dh_sin, dh_cos])?,
        })
    }
}
