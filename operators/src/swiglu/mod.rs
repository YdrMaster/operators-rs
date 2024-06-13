#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use common::{Device, Scheme};
type Params<D> = (
    *mut <D as Device>::Byte,   // gate
    *const <D as Device>::Byte, // up
);

pub trait Swiglu<D: Device>: Scheme<LayoutAttrs = LayoutAttrs, Params = Params<D>> {}
