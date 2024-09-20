﻿use super::{args::SchemeLayout, Args, MatMul};
use crate::common_cpu::Handle as Cpu;
use common::{type_not_support, LaunchError, QueueOf, SchemeError};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl MatMul<Cpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), SchemeError> {
        Ok(())
    }

    fn launch(&self, args: &Self::Args, _queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
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

        let c = c_base as usize;
        let [a, b] = if ab_swap {
            [b_base, a_base]
        } else {
            [a_base, b_base]
        }
        .map(|ptr| ptr as usize);
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        macro_rules! gemm {
            ($ty:ty; $alpha:expr, $beta:expr) => {
                (0..batch as isize).into_par_iter().for_each(|i| unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        (c as *mut $ty).offset(i * c_stride),
                        c_ld,
                        1,
                        beta != 0.,
                        (a as *const $ty).offset(i * a_stride),
                        lhs_cs,
                        lhs_rs,
                        (b as *const $ty).offset(i * b_stride),
                        rhs_cs,
                        rhs_rs,
                        $beta,
                        $alpha,
                        false,
                        false,
                        false,
                        gemm::Parallelism::Rayon(0),
                    )
                })
            };
        }

        use digit_layout::types as ty;
        use gemm::f16;
        match dt {
            ty::F16 => gemm!(f16; f16::from_f32(alpha), f16::from_f32(beta)),
            ty::F32 => gemm!(f32; alpha, beta),
            ty::F64 => gemm!(f64; alpha as _, beta as _),
            _ => Err(type_not_support(format!("Unsupported {dt}")))?,
        }
        Ok(())
    }
}
