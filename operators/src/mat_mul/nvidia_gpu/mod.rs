use super::{args::SchemeLayout, Args, MatMul};
use crate::nvidia_gpu::{Handle as Gpu, Internal as Handle};
use common::{locate_error, ErrorPosition, QueueOf};
use cublas::cublas;
use cuda::AsRaw;
use digit_layout::types::F16;
use half::f16;
use std::{ffi::c_void, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
}

impl MatMul<Gpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
        }
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), Self::SchemeError> {
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
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

        let c = c_base.cast::<c_void>();
        let a = a.cast::<c_void>();
        let b = b.cast::<c_void>();
        let alpha = f16::from_f32(alpha);
        let beta = f16::from_f32(beta);

        self.handle.cublas(queue, |handle| {
            cublas!(cublasGemmStridedBatchedEx(
                handle.as_raw(),
                if a_trans {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                if b_trans {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                m as _,
                n as _,
                k as _,
                ((&alpha) as *const f16).cast(),
                a,
                cudaDataType_t::CUDA_R_16F,
                a_ld as _,
                a_stride as _,
                b,
                cudaDataType_t::CUDA_R_16F,
                b_ld as _,
                b_stride as _,
                ((&beta) as *const f16).cast(),
                c,
                cudaDataType_t::CUDA_R_16F,
                c_ld as _,
                c_stride as _,
                batch as _,
                cublasComputeType_t::CUBLAS_COMPUTE_16F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            ));
        });

        Ok(())
    }
}
