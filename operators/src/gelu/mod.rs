#[cfg(any(use_cpu, test))]
pub mod common_cpu;

mod args;
pub use args::Args;

crate::op_trait!(Gelu);
