use super::{layout::SchemeLayout, LayoutAttrs, Params};
use common::{locate_error, DataLayout, ErrorPosition, QueueOf, F16};
use dev_nvidia_gpu::{
    cuda::{self, ComputeCapability, Ptx},
    Device as Gpu, __global__,
};
use log::warn;
use std::ffi::{c_int, c_uint, CString};

pub struct Config {
    pub data_layout: DataLayout,
    pub max_num_threads_block: usize,
    pub compute_capability: ComputeCapability,
}

impl Config {
    pub fn config_for(dev: &cuda::Device, data_layout: DataLayout) -> Self {
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

const NAME: &str = "rope_f16";

impl common::Operator<Gpu> for Operator {
    type Config = Config;
    type Error = ErrorPosition;
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        let &Self::Config {
            data_layout,
            max_num_threads_block,
            compute_capability,
        } = config;

        const CODE: &str = include_str!("rope.cuh");

        if data_layout != F16 {
            return Err(locate_error!());
        }
        let code = format!(
            r#"{CODE}

    extern "C" __global__ void {NAME}(
        half2 *__restrict__ t,
        int const stride_token,
        int const stride_head,
        unsigned int const *__restrict__ pos,
        float theta
    ){{
        padding(t, stride_token, stride_head, pos, theta);
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
    block: (c_uint, c_uint),
    stride_token: c_int,
    stride_head: c_int,
    offset_t: usize,
    offset_pos: usize,
}

impl common::Scheme<Gpu, Operator> for Scheme {
    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        let SchemeLayout {
            n,
            nh,
            dh,
            stride_token,
            stride_head,
            offset_t,
            offset_pos,
        } = SchemeLayout::new(F16, layout)?;
        let dh = dh / 2;
        let stride_token = (stride_token / 2) as _;
        let stride_head = (stride_head / 2) as _;
        if op.max_num_threads_block % dh != 0 {
            return Err(locate_error!());
        }
        //                    nh == nh_h * nh_l
        // max_num_threads_block >= nh_l x dh
        let max_nh_l = (op.max_num_threads_block / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;
        Ok(Self {
            global: op.global,

            grid: (n as _, nh_h as _),
            block: (nh_l as _, dh as _),
            stride_token,
            stride_head,
            offset_t,
            offset_pos,
        })
    }

    type Params<'ctx> = Params<Gpu>;
    fn launch(&self, params: &Self::Params<'_>, queue: &QueueOf<Gpu>) {
        let name = CString::new(NAME).unwrap();
        let (t, pos, theta) = params;
        let t = unsafe { t.add(self.offset_t) };
        let pos = unsafe { pos.add(self.offset_pos) };
        let params = cuda::params![t, self.stride_token, self.stride_head, pos, theta];
        self.global
            .launch(&name, self.grid, self.block, params.as_ptr(), 0, queue);
    }
}
