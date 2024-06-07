#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod layout;
pub use layout::KnTensorLayout;

pub trait RopeScheme<D: crate::Device> {
    fn launch(&self, t: *mut D::Byte, pos: *const D::Byte, theta: f32);
}
