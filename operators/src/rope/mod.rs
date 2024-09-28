#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
pub use args::Args;

crate::op_trait! { Rope
    /// 生成 sincos 表（[n, 2, dh]）。
    fn build_sincos<QA>(dt: digit_layout::DigitLayout, nctx: usize, dh: usize, queue_alloc: &QA) -> QA::DevMem
        where QA: crate::QueueAlloc<Hardware = Self::Hardware>;
    /// 为多个请求生成位置向量（[nt]）。
    fn build_pos<I, QA>(nt: usize, iter: I, queue_alloc: &QA) -> QA::DevMem
        where I: IntoIterator<Item = Seq>,
              QA: crate::QueueAlloc<Hardware = Self::Hardware>;
}

pub struct Seq {
    pub pos: usize,
    pub len: usize,
}

fn fill_pos<I>(host: &mut [u32], iter: I)
where
    I: IntoIterator<Item = Seq>,
{
    iter.into_iter()
        .flat_map(|seq| seq.pos..seq.pos + seq.len)
        .zip(host)
        .for_each(|(pos, out)| *out = pos as _)
}
