use crate::utils::MutPtr;
use common::{locate_error, ErrorPosition, Handle, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub att_layout: TensorLayout,
    pub att_base: MutPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.att_layout.dt();
        if self.att_layout.ndim() != 3 {
            return Err(locate_error!());
        }
        Ok(Meta { dt })
    }
}
