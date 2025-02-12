#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod cuda;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_cl)]
pub mod opencl;

mod args;
pub use args::{Args, AttnMask};

crate::op_trait!(FusedSoftmax);
