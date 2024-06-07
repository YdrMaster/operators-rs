#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::KnTensorLayout;

pub trait SwigluScheme<D: crate::Device> {
    fn launch(&self, gate: *mut D::Byte, up: *const D::Byte);
}
