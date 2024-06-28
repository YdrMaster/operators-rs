use super::{args::SchemeLayout, Args, MatMul};
use crate::common_cpu::Handle as Cpu;
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
use half::f16;

pub struct Operator;

impl MatMul<Cpu> for Operator {}

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
        let SchemeLayout {
            dt,
            ab_swap,
            a_trans,
            b_trans,
            batch,
            m,
            n,
            k,
            c_stride,
            c_ld,
            a_stride,
            a_ld,
            b_stride,
            b_ld,
        } = args.layout()?;
        let &Args {
            c_base,
            beta,
            a_base,
            b_base,
            alpha,
            ..
        } = args;

        if dt != F16 {
            return Err(locate_error!());
        }

        let (a, b) = if ab_swap {
            (b_base, a_base)
        } else {
            (a_base, b_base)
        };
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        let c = c_base.cast::<f16>();
        let a = a.cast::<f16>();
        let b = b.cast::<f16>();

        for i in 0..batch as isize {
            unsafe {
                let c = c.offset(i * c_stride);
                let a = a.offset(i * a_stride);
                let b = b.offset(i * b_stride);
                gemm::gemm(
                    m,
                    n,
                    k,
                    c,
                    c_ld,
                    1,
                    beta != 0.,
                    a,
                    lhs_cs,
                    lhs_rs,
                    b,
                    rhs_cs,
                    rhs_rs,
                    f16::from_f32(beta),
                    f16::from_f32(alpha),
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                )
            }
        }
        Ok(())
    }
}
