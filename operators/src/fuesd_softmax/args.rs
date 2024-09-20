use crate::utils::MutPtr;
use common::{rank_not_support, Handle, ParamError, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub att_layout: TensorLayout,
    pub att_base: MutPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let dt = self.att_layout.dt();
        if self.att_layout.ndim() != 3 {
            return Err(rank_not_support(""));
        }
        Ok(Meta { dt })
    }
}
