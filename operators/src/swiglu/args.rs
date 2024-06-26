use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
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
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.gate_layout.dt();
        if self.up_layout.dt() != dt {
            return Err(locate_error!());
        }

        let &[gn, gd] = self.gate_layout.shape() else {
            return Err(locate_error!());
        };
        let &[un, ud] = self.up_layout.shape() else {
            return Err(locate_error!());
        };
        let Ok(&n) = Argument::merge(&[gn, un]) else {
            return Err(locate_error!());
        };
        let Ok(&d) = Argument::merge(&[gd, ud]) else {
            return Err(locate_error!());
        };

        Ok(Meta { dt, n, d })
    }
}
