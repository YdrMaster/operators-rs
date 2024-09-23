#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
mod kv_pair;

pub use args::{Args, SampleArgs, SampleArgsError};
pub use kv_pair::KVPair;

crate::op_trait! { RandomSample
    fn build_indices<QA>(n:usize, queue_alloc: &QA) -> QA::DevMem
        where QA: crate::QueueAlloc<Hardware = Self::Hardware>;
}
