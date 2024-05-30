mod layout;

#[cfg(feature = "common-cpu")]
mod common_cpu;
#[cfg(feature = "nv-gpu")]
mod nv_gpu;

use crate::{DataLayout, Device};

pub use layout::RmsNormTensorLayout;

pub struct RmsNorm {
    _dt: DataLayout,
}

pub trait RmsNormScheme<D: Device> {
    fn launch(&self, y: *mut D::Byte, x: *const D::Byte, w: *const D::Byte, epsilon: f32);
}
