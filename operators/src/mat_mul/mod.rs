#[cfg(use_cpu)]
pub mod common_cpu;
// #[cfg(use_cuda)]
// pub mod nvidia_gpu;

mod layout;
pub use layout::LayoutAttrs;

use common::{Device, Scheme};
type Params<D> = (
    *mut <D as Device>::Byte,   // c
    f32,                        // β
    *const <D as Device>::Byte, // a
    *const <D as Device>::Byte, // b
    f32,                        // α
);

pub trait MatMul<D: Device>: Scheme<LayoutAttrs = LayoutAttrs, Params = Params<D>> {}
