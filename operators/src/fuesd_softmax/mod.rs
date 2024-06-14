﻿#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use crate::utils::*;
op_trait!(FuesdSoftmax);
type Params<D> = MutPtr<D>; // att