﻿#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
pub use args::Args;

crate::utils::op_trait!(Reform);
