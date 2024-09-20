use super::{args::SchemeLayout, Args, MatMul};
use crate::nvidia_gpu::{Handle as Gpu, Internal as Handle};
use common::{type_not_support, LaunchError, QueueOf, SchemeError};
use cublas::cublas;
use dev_mempool::cuda::AsRaw;
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

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
        }
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), SchemeError> {
        Ok(())
    }

    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
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
            return Err(type_not_support("").into());
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

#[cfg(test)]
mod test {
    use super::Args;
    use common::{Handle, TensorLayout};
    use digit_layout::DigitLayout;

    const ALPHA: f32 = 0.5;
    const BETA: f32 = 1.;

    fn args<H: Handle>(
        dt: DigitLayout,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[batch, m, n]),
            c_base,
            beta: BETA,
            a_layout: TensorLayout::new_contiguous(dt, &[batch, m, k]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[batch, k, n]),
            b_base,
            alpha: ALPHA,
        }
    }

    #[test]
    fn test_compute() {
        use super::{super::common_cpu::Operator as RefOp, Gpu, Operator};
        use crate::common_cpu::{Handle as Cpu, ThisThread};
        use crate::{
            nvidia_gpu::cast_load,
            utils::{Diff, ErrorCollector},
        };
        use common::Operator as _;
        use dev_mempool::cuda::memcpy_d2h;
        use digit_layout::types::{F16, F64};
        use half::f16;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let cpu_op = RefOp::new(&Cpu);
        let gpu_op = Operator::new(&gpu);

        let batch = 4;
        let k = 2048;
        let n = 5632;
        for m in [1, 7, 64, 255, 1024] {
            let mut a = vec![0.0f64; batch * m * k];
            let mut b = vec![0.0f64; batch * k * n];
            let mut c = vec![0.0f64; batch * m * n];
            rand::thread_rng().fill(&mut a[..]);
            rand::thread_rng().fill(&mut b[..]);
            rand::thread_rng().fill(&mut c[..]);
            let a = a;
            let b = b;

            let c_ans = gpu.apply(|ctx| {
                let stream = ctx.stream();
                let mut c = cast_load(&c, |&x| f16::from_f64(x), &stream);
                let a = cast_load(&a, |&x| f16::from_f64(x), &stream);
                let b = cast_load(&b, |&x| f16::from_f64(x), &stream);

                gpu_op
                    .launch(
                        &args(
                            F16,
                            batch,
                            m,
                            n,
                            k,
                            c.as_mut_ptr().cast(),
                            a.as_ptr().cast(),
                            b.as_ptr().cast(),
                        ),
                        &stream,
                    )
                    .unwrap();

                let mut ans = vec![f16::ZERO; batch * m * n];
                memcpy_d2h(&mut ans, &c);
                ans
            });

            let mut c_ref = c;
            cpu_op
                .launch(
                    &args(
                        F64,
                        batch,
                        m,
                        n,
                        k,
                        c_ref.as_mut_ptr().cast(),
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                    ),
                    &ThisThread,
                )
                .unwrap();

            let diff = c_ref
                .into_par_iter()
                .zip(c_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 5e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
