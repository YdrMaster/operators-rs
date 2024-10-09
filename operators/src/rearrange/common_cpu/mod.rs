use super::{args::Scheme, Args, Rearrange};
use crate::{common_cpu::Cpu, ByteOf, LaunchError, QueueAlloc, SchemeError};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl Rearrange<Cpu> for Operator {}

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
        let scheme = Scheme::new(args)?;
        let unit = scheme.unit();
        if scheme.count() == 1 {
            unsafe { std::ptr::copy_nonoverlapping(args.src_base, args.dst_base, unit) };
        } else {
            let dst = args.dst_base as isize;
            let src = args.src_base as isize;
            let idx_strides = scheme.idx_strides();
            let dst_strides = scheme.dst_strides();
            let src_strides = scheme.src_strides();
            (0..scheme.count() as isize)
                .into_par_iter()
                .for_each(|mut rem| {
                    let mut dst = dst;
                    let mut src = src;
                    for (i, &s) in idx_strides.iter().enumerate() {
                        let k = rem / s;
                        dst += k * dst_strides[i];
                        src += k * src_strides[i];
                        rem %= s;
                    }
                    unsafe { std::ptr::copy_nonoverlapping::<u8>(src as _, dst as _, unit) };
                });
        }
        Ok(())
    }
}
