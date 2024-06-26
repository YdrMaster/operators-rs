use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
use digit_layout::{types::U32, DigitLayout};

pub struct Args<H: Handle> {
    pub t_layout: TensorLayout,
    pub t_base: MutPtr<H>,
    pub p_layout: TensorLayout,
    pub p_base: ConstPtr<H>,
    pub theta: f32,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.t_layout.dt();
        if self.p_layout.dt() != U32 {
            return Err(locate_error!());
        }
        if self.t_layout.ndim() != 3 || self.p_layout.ndim() != 1 {
            return Err(locate_error!());
        }

        let &[nt, _, _] = self.t_layout.shape() else {
            unreachable!()
        };
        let &[np] = self.p_layout.shape() else {
            unreachable!()
        };
        let Ok(&n) = Argument::merge(&[nt, np]) else {
            return Err(locate_error!());
        };

        Ok(Meta { dt, n })
    }
}
