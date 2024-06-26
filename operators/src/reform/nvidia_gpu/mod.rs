use super::{layout::SchemeLayout, LayoutAttrs, Params, Reform};
use common::{locate_error, ErrorPosition, QueueOf};
use dev_nvidia_gpu::{
    cuda::{self, ComputeCapability, Ptx},
    Device as Gpu, __global__,
};
use log::warn;
use std::ffi::{c_int, c_uint, CString};

pub struct Config {
    pub num_threads_warp: usize,
    pub max_num_threads_block: usize,
    pub compute_capability: ComputeCapability,
}

impl Config {
    pub fn config_for(dev: &cuda::Device) -> Self {
        Self {
            num_threads_warp: 32,
            max_num_threads_block: dev.max_block_dims().0,
            compute_capability: dev.compute_capability(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operator {
    global: __global__,
    pub num_threads_warp: usize,
    pub max_num_threads_block: usize,
}

const NAME: &str = "reform";

impl common::Operator for Operator {
    type Handle = Gpu;

    type Config = Config;
    type Error = ErrorPosition;
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        let &Self::Config {
            num_threads_warp,
            max_num_threads_block,
            compute_capability,
        } = config;

        const CODE: &str = include_str!("reform.cuh");

        let code = format!(
            r#"{CODE}

extern "C" __global__ void {NAME}(
    void       *__restrict__ dst,
    unsigned int const rsa,
    unsigned int const csa,
    void const *__restrict__ src,
    unsigned int const rsb,
    unsigned int const csb,
    unsigned int const ncols,
    unsigned int const bytes_per_thread
){{
    switch (bytes_per_thread) {{
        case  1: reform<uchar1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  2: reform<uchar2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  4: reform<float1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  8: reform<float2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 16: reform<float4 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 32: reform<double4>(dst, rsa, csa, src, rsb, csb, ncols); break;
    }}
}}
"#
        );

        let (ptx, log) = Ptx::compile(code, compute_capability);
        let ptx = ptx
            .map_err(|e| locate_error!(format!("Failed to compile fused_softmax: {e:?}\n{log}")))?;
        if !log.is_empty() {
            warn!("{log}");
        }
        Ok(Self {
            global: __global__::load(&ptx),
            num_threads_warp,
            max_num_threads_block,
        })
    }
}

pub struct Scheme {
    global: __global__,

    grid: (c_uint, c_uint),
    block: (c_uint, c_uint),
    dst_rs: c_int,
    dst_cs: c_int,
    src_rs: c_int,
    src_cs: c_int,
    c: c_uint,
    bytes_per_thread: usize,
    dst_offset: usize,
    src_offset: usize,
}

impl Reform<Gpu> for Scheme {}

impl common::Scheme for Scheme {
    type Device = Gpu;
    type Operator = Operator;

    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        let SchemeLayout {
            r,
            c,
            z,
            dst_rs,
            dst_cs,
            src_rs,
            src_cs,
            dst_offset,
            src_offset,
        } = SchemeLayout::new(layout)?;

        if z % op.num_threads_warp != 0 {
            return Err(locate_error!());
        }
        let bytes_per_thread = z / op.num_threads_warp;
        if bytes_per_thread > 32 || !bytes_per_thread.is_power_of_two() {
            return Err(locate_error!());
        }

        let max_warp_per_block = op.max_num_threads_block / op.num_threads_warp;
        let grid = (r, (c + max_warp_per_block - 1) / max_warp_per_block);
        let block = ((c + grid.1 - 1) / grid.1, op.num_threads_warp);

        Ok(Self {
            global: op.global,
            grid: (grid.0 as _, grid.1 as _),
            block: (block.0 as _, block.1 as _),
            dst_rs: (dst_rs / z as isize) as _,
            dst_cs: (dst_cs / z as isize) as _,
            src_rs: (src_rs / z as isize) as _,
            src_cs: (src_cs / z as isize) as _,
            c: c as _,
            bytes_per_thread,
            dst_offset,
            src_offset,
        })
    }

    type Params = Params<Gpu>;
    fn launch(&self, params: &Self::Params, queue: &QueueOf<Gpu>) {
        let name = CString::new(NAME).unwrap();
        let &(dst, src) = params;
        let dst = unsafe { dst.add(self.dst_offset) };
        let src = unsafe { src.add(self.src_offset) };
        let params = cuda::params![
            dst,
            self.dst_rs,
            self.dst_cs,
            src,
            self.src_rs,
            self.src_cs,
            self.c,
            self.bytes_per_thread
        ];
        self.global
            .launch(&name, self.grid, self.block, params.as_ptr(), 0, queue);
    }
}
