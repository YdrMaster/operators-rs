use crate::MutPtr;
use crate::{rank_not_support, Hardware, ParamError, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub att_layout: TensorLayout,
    pub att_base: MutPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let dt = self.att_layout.dt();
        if self.att_layout.ndim() != 3 {
            return Err(rank_not_support(""));
        }
        Ok(Meta { dt })
    }
}
