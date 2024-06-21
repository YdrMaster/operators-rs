use super::{super::gcd, layout::SchemeLayout, LayoutAttrs, Params, Swiglu};
use common::{locate_error, ErrorPosition, QueueOf};
use dev_nvidia_gpu::{
    cuda::{self, ComputeCapability, Ptx},
    Device as Gpu, __global__,
};
use digit_layout::{types::F16, DigitLayout};
use log::warn;
use std::ffi::{c_int, c_uint, CString};

pub struct Config {
    pub data_layout: DigitLayout,
    pub max_num_threads_block: usize,
    pub compute_capability: ComputeCapability,
}

impl Config {
    pub fn config_for(dev: &cuda::Device, data_layout: DigitLayout) -> Self {
        Self {
            data_layout,
            max_num_threads_block: dev.max_block_dims().0,
            compute_capability: dev.compute_capability(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operator {
    global: __global__,
    max_num_threads_block: usize,
}

const NAME: &str = "swiglu_f16";

impl common::Operator for Operator {
    type Device = Gpu;

    type Config = Config;
    type Error = ErrorPosition;
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        let &Self::Config {
            data_layout,
            max_num_threads_block,
            compute_capability,
        } = config;

        const CODE: &str = include_str!("swiglu.cuh");

        if data_layout != F16 {
            return Err(locate_error!());
        }
        let code = format!(
            r#"{CODE}

        extern "C" __global__ void {NAME}(
            half *__restrict__ gate,
            int const stride_gate,
            half const *__restrict__ up,
            int const stride_up
        ){{
            swiglu(gate, stride_gate, up, stride_up);
        }}"#
        );

        let (ptx, log) = Ptx::compile(code, compute_capability);
        let ptx =
            ptx.map_err(|e| locate_error!(format!("Failed to compile {NAME}: {e:?}\n{log}")))?;
        if !log.is_empty() {
            warn!("{log}");
        }
        Ok(Self {
            global: __global__::load(&ptx),
            max_num_threads_block,
        })
    }
}

pub struct Scheme {
    global: __global__,

    grid: (c_uint, c_uint),
    block: c_uint,
    stride_gate: c_int,
    stride_up: c_int,
    offset_gate: usize,
    offset_up: usize,
}

impl Swiglu<Gpu> for Scheme {}

impl common::Scheme for Scheme {
    type Device = Gpu;
    type Operator = Operator;

    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        let SchemeLayout {
            n,
            d,
            stride_gate,
            stride_up,
            offset_gate,
            offset_up,
        } = SchemeLayout::new(F16, layout)?;
        let block = gcd(op.max_num_threads_block, d);
        Ok(Self {
            global: op.global,

            grid: (n as _, (d / block) as _),
            block: block as _,
            stride_gate: stride_gate as _,
            stride_up: stride_up as _,
            offset_gate,
            offset_up,
        })
    }

    type Params = Params<Gpu>;
    fn launch(&self, params: &Self::Params, queue: &QueueOf<Gpu>) {
        let (gate, up) = params;
        let name = CString::new(NAME).unwrap();
        let gate = unsafe { gate.add(self.offset_gate) };
        let up = unsafe { up.add(self.offset_up) };
        let params = cuda::params![gate, self.stride_gate, up, self.stride_up];
        self.global
            .launch(&name, self.grid, self.block, params.as_ptr(), 0, queue);
    }
}
