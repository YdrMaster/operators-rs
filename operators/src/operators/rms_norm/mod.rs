#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::KnTensorLayout;

pub trait RmsNormScheme<D: crate::Device> {
    fn launch(&self, y: *mut D::Byte, x: *const D::Byte, w: *const D::Byte, epsilon: f32);
}
