use super::{layout::SchemeLayout, LayoutAttrs, MatMul, Params};
use common::{locate_error, DataLayout, ErrorPosition, QueueOf, F16};
use dev_nvidia_gpu::{
    cublas::cublas,
    cuda::{self, bindings::CUdevice, AsRaw},
    preload_cublas, use_cublas, Device as Gpu,
};
use half::f16;
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Debug)]
pub struct Operator {
    dt: DataLayout,
}

impl common::Operator for Operator {
    type Device = Gpu;

    type Config = DataLayout;
    type Error = ErrorPosition;
    #[inline]
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        if *config == F16 {
            preload_cublas();
            Ok(Self { dt: *config })
        } else {
            Err(locate_error!())
        }
    }
}

pub struct Scheme(SchemeLayout);

impl MatMul<Gpu> for Scheme {}

impl common::Scheme for Scheme {
    type Device = Gpu;
    type Operator = Operator;

    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    #[inline]
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        SchemeLayout::new(op.dt, layout).map(Self)
    }

    type Params = Params<Gpu>;
    fn launch(&self, params: &Self::Params, stream: &QueueOf<Gpu>) {
        if crate::is_recording() {
            record_scheme(unsafe { stream.ctx().dev().as_raw() }, &self.0);
        }

        let SchemeLayout {
            batch,
            m,
            n,
            k,

            c_stride,
            c_offset,
            c_ld,
            ab_swap,

            a_stride,
            a_offset,
            a_ld,
            a_trans,

            b_stride,
            b_offset,
            b_ld,
            b_trans,
        } = self.0;
        let &(c, beta, a, b, alpha) = params;
        let (a, b) = if ab_swap { (b, a) } else { (a, b) };

        let c = unsafe { c.add(c_offset).cast::<c_void>() };
        let a = unsafe { a.add(a_offset).cast::<c_void>() };
        let b = unsafe { b.add(b_offset).cast::<c_void>() };
        let alpha = f16::from_f32(alpha);
        let beta = f16::from_f32(beta);

        use_cublas(stream, |handle| {
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
        })
    }
}

#[inline(always)]
fn record() -> &'static Mutex<HashMap<SchemeLayout, Vec<usize>>> {
    static RECORD: OnceLock<Mutex<HashMap<SchemeLayout, Vec<usize>>>> = OnceLock::new();
    RECORD.get_or_init(Default::default)
}

fn record_scheme(dev: CUdevice, scheme: &SchemeLayout) {
    let mut map = record().lock().unwrap();
    let vec = map
        .entry(scheme.clone())
        .or_insert_with(|| vec![0; cuda::Device::count() as _]);
    vec[dev as usize] += 1;
}

pub fn get_record() -> Vec<(SchemeLayout, Vec<usize>)> {
    let map = record().lock().unwrap();
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}
