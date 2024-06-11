#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use crate::Device;
type Params<D> = (
    *mut <D as Device>::Byte,   // t
    *const <D as Device>::Byte, // pos
    f32,                        // θ
);
