#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use common::Device;
type Params<D> = (
    *mut <D as Device>::Byte,   // y
    *const <D as Device>::Byte, // x
    *const <D as Device>::Byte, // w
    f32,                        // ε
);
