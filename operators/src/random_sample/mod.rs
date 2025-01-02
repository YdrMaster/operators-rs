#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod cuda;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_cl)]
pub mod opencl;

mod args;
mod kv_pair;

pub use args::{Args, SampleArgs, SampleArgsError};
pub use kv_pair::KVPair;

crate::op_trait! { RandomSample
    fn build_indices<QA>(n: usize, queue_alloc: &QA) -> Indices<QA::DevMem>
        where QA: crate::QueueAlloc<Hardware = Self::Hardware>;
}

pub struct Indices<Mem> {
    pub n: usize,
    pub mem: Mem,
}
