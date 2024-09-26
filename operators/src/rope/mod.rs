#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
pub use args::Args;

crate::op_trait! { Rope
    fn build_sincos<QA>(n: usize, queue_alloc: &QA) -> QA::DevMem
        where QA: crate::QueueAlloc<Hardware = Self::Hardware>;

    fn build_pos<I, QA>(nt:usize, iter: I, queue_alloc: &QA) -> QA::DevMem
        where I: IntoIterator<Item = Seq>,
              QA: crate::QueueAlloc<Hardware = Self::Hardware>;
}

pub struct Seq {
    pub pos: usize,
    pub len: usize,
}
