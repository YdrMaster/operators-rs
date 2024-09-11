use super::{args::SchemeLayout, Args, MatMul};
use crate::common_cpu::Handle as Cpu;
use common::{locate_error, ErrorPosition, QueueOf};

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

        let (a, b) = if ab_swap {
            (b_base, a_base)
        } else {
            (a_base, b_base)
        };
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        macro_rules! gemm {
            ($c:expr, $beta:expr,$a:expr,$b:expr,$alpha:expr) => {
                for i in 0..batch as isize {
                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            $c.offset(i * c_stride),
                            c_ld,
                            1,
                            beta != 0.,
                            $a.offset(i * a_stride),
                            lhs_cs,
                            lhs_rs,
                            $b.offset(i * b_stride),
                            rhs_cs,
                            rhs_rs,
                            $beta,
                            $alpha,
                            false,
                            false,
                            false,
                            gemm::Parallelism::Rayon(0),
                        )
                    }
                }
            };
        }

        use digit_layout::types as ty;
        match dt {
            ty::F16 => {
                use gemm::f16;
                let c = c_base.cast::<f16>();
                let a = a.cast::<f16>();
                let b = b.cast::<f16>();
                let alpha = f16::from_f32(alpha);
                let beta = f16::from_f32(beta);
                gemm!(c, beta, a, b, alpha);
            }
            ty::F32 => {
                let c = c_base.cast::<f32>();
                let a = a.cast::<f32>();
                let b = b.cast::<f32>();
                gemm!(c, beta, a, b, alpha);
            }
            ty::F64 => {
                let c = c_base.cast::<f64>();
                let a = a.cast::<f64>();
                let b = b.cast::<f64>();
                let alpha = alpha as _;
                let beta = beta as _;
                gemm!(c, beta, a, b, alpha);
            }
            _ => return Err(locate_error!("Unsupported {dt}")),
        }

        Ok(())
    }
}
