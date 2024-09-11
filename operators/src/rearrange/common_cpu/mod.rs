use super::{args::Scheme, Args, Rearrange};
use crate::common_cpu::Handle as Cpu;
use common::{ErrorPosition, QueueOf};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl Rearrange<Cpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), Self::SchemeError> {
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        _queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
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
