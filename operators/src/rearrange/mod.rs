#[cfg(use_ascend)]
pub mod ascend;
#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
#[cfg(use_cl)]
pub mod opencl;

mod args;
pub use args::Args;

crate::op_trait!(Rearrange);
