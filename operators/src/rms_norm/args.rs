use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
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
    pub dt: DigitLayout,
    pub n: Argument<usize>,
    pub d: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.y_layout.dt();
        if self.x_layout.dt() != dt || self.w_layout.dt() != dt {
            return Err(locate_error!());
        }
        if self.y_layout.ndim() != 2 || self.x_layout.ndim() != 2 || self.w_layout.ndim() != 1 {
            return Err(locate_error!());
        }

        let &[ny, dy] = self.y_layout.shape() else {
            unreachable!()
        };
        let &[nx, dx] = self.x_layout.shape() else {
            unreachable!()
        };
        let &[dw] = self.w_layout.shape() else {
            unreachable!()
        };
        let Ok(&n) = Argument::merge(&[ny, nx]) else {
            return Err(locate_error!());
        };
        let Ok(&d) = Argument::merge(&[dy, dx, dw]) else {
            return Err(locate_error!());
        };

        Ok(Meta { dt, n, d })
    }
}
