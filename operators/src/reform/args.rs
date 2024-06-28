use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
use digit_layout::DigitLayout;
use std::iter::zip;

pub struct Args<H: Handle> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.dst_layout.dt();
        if self.src_layout.dt() != dt {
            return Err(locate_error!());
        }
        let ndim = self.dst_layout.ndim();
        if ndim < 2 || self.src_layout.ndim() != ndim {
            return Err(locate_error!());
        }
        for (&dst, &src) in zip(self.dst_layout.shape(), self.src_layout.shape()) {
            if Argument::merge(&[dst, src]).is_err() {
                return Err(locate_error!());
            }
        }
        Ok(Meta { dt })
    }
}
