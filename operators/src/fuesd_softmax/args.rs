use crate::{rank_not_support, Hardware, MutPtr, SchemeError, TensorLayout};
use digit_layout::DigitLayout;
use std::ptr::null_mut;

pub struct Args<H: Hardware> {
    pub att_mask: AttnMask,
    pub att_layout: TensorLayout,
    pub att_base: MutPtr<H>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(att_mask: AttnMask, att_layout: TensorLayout) -> Self {
        Self {
            att_mask,
            att_layout,
            att_base: null_mut(),
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let dt = self.att_layout.dt();
        if self.att_layout.ndim() != 3 {
            return Err(rank_not_support(""));
        }
        Ok(Meta { dt })
    }
}
