use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub t_layout: TensorLayout,
    pub t_base: MutPtr<H>,
    pub p_layout: TensorLayout,
    pub p_base: ConstPtr<H>,
    pub theta: f32,
}

pub(super) struct Meta {
    pub dt_t: DigitLayout,
    pub dt_p: DigitLayout,
    pub n: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        use digit_layout::LayoutContent::{Real, Unsigned};
        let dt_t = self.t_layout.dt();
        let dt_p = self.p_layout.dt();
        // tokens must be floating-point numbers
        if !matches!(dt_t.decode(), Real { exponent: 1.., .. },) {
            return Err(locate_error!());
        }
        // positions must be unsigned integers
        if !matches!(dt_p.decode(), Unsigned { .. }) {
            return Err(locate_error!());
        }
        let &[nt, _, _] = self.t_layout.shape() else {
            return Err(locate_error!());
        };
        let &[np] = self.p_layout.shape() else {
            return Err(locate_error!());
        };
        let Ok(&n) = Argument::merge(&[nt, np]) else {
            return Err(locate_error!());
        };
        Ok(Meta { dt_t, dt_p, n })
    }
}
