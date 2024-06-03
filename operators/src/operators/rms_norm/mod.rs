mod layout;

#[cfg(detected_cpu)]
pub mod common_cpu;
#[cfg(detected_cuda)]
pub mod nvidia_gpu;

pub use layout::RmsNormTensorLayout;

pub trait RmsNormScheme<D: crate::Device> {
    fn launch(&self, y: *mut D::Byte, x: *const D::Byte, w: *const D::Byte, epsilon: f32);
}
