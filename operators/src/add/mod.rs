//! c = a + b

#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod cuda;

mod args;
pub use args::Args;

crate::op_trait!(Add);
