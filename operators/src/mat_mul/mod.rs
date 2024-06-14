#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use crate::utils::*;
op_trait!(MatMul);
type Params<D> = (
    MutPtr<D>,   // c
    f32,         // β
    ConstPtr<D>, // a
    ConstPtr<D>, // b
    f32,         // α
);
