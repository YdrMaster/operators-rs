#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
pub use args::{Args, KVpair, KV_PAIR};

crate::utils::op_trait!(ArgMax);
