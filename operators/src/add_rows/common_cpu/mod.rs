use super::{args::Meta, AddRows, Args};
use crate::{common_cpu::Cpu, get_static, ByteOf, LaunchError, QueueAlloc, SchemeError, Unsigned};
use digit_layout::types as ty;
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::AddAssign;

pub struct Operator;

impl AddRows<Cpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

    fn new(_node: &Self::TopoNode) -> Self {
        Self
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta {
            dt,
            dt_idx,
            batch: b,
            m,
            n,
            k,
        } = args.meta()?;
        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
            idx_layout,
            idx_base,
        } = args;

        let &[bsd, msd, nsd] = dst_layout.strides() else {
            unreachable!()
        };
        let &[kss, nss] = src_layout.strides() else {
            unreachable!()
        };
        let &[bsi, msi] = idx_layout.strides() else {
            unreachable!()
        };

        get_static! {
            b   m   n   k
            bsd msd nsd
            bsi msi nss kss
        }

        let dst = *dst_base as usize;
        let src = *src_base as usize;
        let idx = *idx_base as usize;

        macro_rules! calculate {
            ($t:ty, $i:ty) => {
                (0..b * m).into_par_iter().for_each(|bm| {
                    Scheme::<$t, $i> {
                        dst: dst as _,
                        src: src as _,
                        idx: idx as _,
                        m,
                        n,
                        k,
                        bsd,
                        msd,
                        nsd,
                        kss,
                        nss,
                        bsi,
                        msi,
                    }
                    .calculate(bm)
                })
            };
        }

        match (dt, dt_idx) {
            (ty::F16, ty::U32) => calculate!(f16, u32),
            (ty::F32, ty::U32) => calculate!(f32, u32),
            (ty::F64, ty::U32) => calculate!(f64, u32),
            (ty::F16, ty::U64) => calculate!(f16, u64),
            (ty::F32, ty::U64) => calculate!(f32, u64),
            (ty::F64, ty::U64) => calculate!(f64, u64),
            (_, _) => todo!(),
        }
        Ok(())
    }
}

struct Scheme<T, I> {
    dst: *mut T,
    src: *const T,
    idx: *const I,
    m: usize,
    n: usize,
    k: usize,
    bsd: isize,
    msd: isize,
    nsd: isize,
    kss: isize,
    nss: isize,
    bsi: isize,
    msi: isize,
}

impl<T, I> Scheme<T, I>
where
    T: AddAssign + Copy,
    I: Unsigned + Copy,
{
    fn calculate(&self, bm: usize) {
        let b = (bm / self.m) as isize;
        let m = (bm % self.m) as isize;
        let dst = unsafe { self.dst.byte_offset(b * self.bsd + m * self.msd) };
        let idx = unsafe { *self.idx.byte_offset(b * self.bsi + m * self.msi) }.val();
        assert!(idx < self.k);

        let src = unsafe { self.src.byte_offset(idx as isize * self.kss) };
        for i in 0..self.n as isize {
            unsafe {
                let dst = dst.byte_offset(i * self.nsd);
                let src = src.byte_offset(i * self.nss);
                *dst += *src;
            }
        }
    }
}
