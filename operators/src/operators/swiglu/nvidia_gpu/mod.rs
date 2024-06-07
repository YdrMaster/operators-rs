use std::{
    ffi::{c_int, c_uint, CString},
    sync::Arc,
};

use crate::{
    devices::nvidia_gpu::Device as Gpu, locate_error, operators::gcd, DataLayout, Device,
    ErrorPosition, F16,
};
use cuda::{ComputeCapability, ContextResource, ContextSpore, Ptx, StreamSpore};

use super::{layout::SchemeLayout, KnTensorLayout, SwigluScheme};

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

pub struct Operator {
    ptx: Ptx,
    max_num_threads_block: usize,
}

const NAME: &str = "swiglu_f16";

impl crate::Operator<Gpu> for Operator {
    type Config = Config;
    type ConfigError = ErrorPosition;

    fn config(
        Self::Config {
            data_layout,
            max_num_threads_block,
            compute_capability,
        }: Self::Config,
    ) -> Result<Self, Self::ConfigError> {
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
            ptx,
            max_num_threads_block,
        })
    }

    type Kernel = Kernel;
    type LoadError = ();

    fn load(&self, ctx: &<Gpu as Device>::Context) -> Result<Self::Kernel, Self::LoadError> {
        Ok(Kernel {
            context: ctx.clone(),
            module: Arc::new(ctx.apply(|ctx| ctx.load(&self.ptx).sporulate())),
            max_num_threads_block: self.max_num_threads_block,
        })
    }
}

pub struct Kernel {
    context: Arc<cuda::Context>,
    module: Arc<cuda::ModuleSpore>,
    max_num_threads_block: usize,
}

impl crate::Kernel<Gpu> for Kernel {
    type Scheme = Scheme;
    type Config = (KnTensorLayout, StreamSpore);
    type SchemeError = ErrorPosition;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError> {
        let (layout, stream) = config;
        let layout = SchemeLayout::new(F16, layout)?;
        let block = gcd(self.max_num_threads_block, layout.d);
        Ok(Self::Scheme {
            context: self.context.clone(),
            module: self.module.clone(),
            stream,

            grid: (layout.n as _, (layout.d / block) as _),
            block: block as _,
            stride_gate: layout.stride_gate as _,
            stride_up: layout.stride_up as _,
        })
    }
}

pub struct Scheme {
    context: Arc<cuda::Context>,
    module: Arc<cuda::ModuleSpore>,
    stream: StreamSpore,

    grid: (c_uint, c_uint),
    block: c_uint,
    stride_gate: c_int,
    stride_up: c_int,
}

impl SwigluScheme<Gpu> for Scheme {
    fn launch(&self, gate: *mut <Gpu as Device>::Byte, up: *const <Gpu as Device>::Byte) {
        let name = CString::new(NAME).unwrap();
        self.context.apply(|ctx| {
            let stream = self.stream.sprout_ref(ctx);
            let module = self.module.sprout_ref(ctx);
            let kernel = module.get_kernel(&name);
            let params = cuda::params![gate, self.stride_gate, up, self.stride_up];
            kernel.launch(self.grid, self.block, params.as_ptr(), 0, Some(&stream));
        });
    }
}
