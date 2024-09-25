#[cfg(use_neuware)]
pub mod cambricon_mlu;
#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
pub use args::Args;

crate::op_trait!(Swiglu);
