use crate::{utils::rank_error, Hardware, LaunchError, MutPtr, TensorLayout};
use digit_layout::DigitLayout;

pub struct Args<H: Hardware> {
    pub layout: TensorLayout,
    pub base: MutPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: usize,
    pub d: usize,
}

impl<H: Hardware> Args<H> {
    pub fn new_layout(layout: TensorLayout) -> Self {
        use std::ptr::null_mut;
        Self {
            layout,
            base: null_mut(),
        }
    }

    pub(super) fn meta(&self) -> Result<Meta, LaunchError> {
        let Self { layout, .. } = self;

        let &[n, d] = &*layout.shape() else {
            return Err(rank_error("layout", 2, layout.ndim()));
        };

        Ok(Meta {
            dt: layout.dt,
            n,
            d,
        })
    }
}
