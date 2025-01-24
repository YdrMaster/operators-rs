#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod cuda;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_cl)]
pub mod opencl;

mod args;
pub use args::Args;

crate::op_trait! { Rope
    /// 生成 sincos 表（[2, n, dh]）。
    fn build_sincos<QA>(dt: digit_layout::DigitLayout, nctx: usize, dh: usize, queue_alloc: &QA) -> SinCosTable<QA::DevMem>
        where QA: crate::QueueAlloc<Hardware = Self::Hardware>;
    /// 为多个请求生成位置向量（[nt]）。
    fn build_pos<I, QA>(dt: digit_layout::DigitLayout, nt: usize, iter: I, queue_alloc: &QA) -> QA::DevMem
        where I: IntoIterator<Item = Seq>,
              QA: crate::QueueAlloc<Hardware = Self::Hardware>;
}

pub struct Seq {
    pub pos: usize,
    pub len: usize,
}

pub struct SinCosTable<Mem> {
    pub nctx: usize,
    pub mem: Mem,
}

trait PosTy {
    fn from_usize(p: usize) -> Self;
}

impl PosTy for u32 {
    fn from_usize(p: usize) -> Self {
        p as _
    }
}

impl PosTy for u64 {
    fn from_usize(p: usize) -> Self {
        p as _
    }
}

fn fill_pos<T, I>(ptr: *mut T, len: usize, iter: I)
where
    T: PosTy,
    I: IntoIterator<Item = Seq>,
{
    iter.into_iter()
        .flat_map(|seq| seq.pos..seq.pos + seq.len)
        .zip(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
        .for_each(|(pos, out)| *out = T::from_usize(pos))
}
