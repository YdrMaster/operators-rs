// #[cfg(use_cpu)]
// pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use common::{Device, Scheme};
type Params<D> = (
    *mut <D as Device>::Byte,   // y
    *const <D as Device>::Byte, // x
);

pub trait Reform<D: Device>: Scheme<LayoutAttrs = LayoutAttrs, Params = Params<D>> {}
