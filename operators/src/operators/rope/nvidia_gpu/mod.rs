use super::{layout::SchemeLayout, KnTensorLayout, RopeScheme};
use crate::{
    devices::nvidia_gpu::Device as Gpu, locate_error, DataLayout, Device, ErrorPosition, F16,
};
use cuda::{ComputeCapability, ContextResource, ContextSpore, Ptx, StreamSpore};
use std::{
    ffi::{c_int, c_uint, CString},
    sync::Arc,
};

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

const NAME: &str = "rope_f16";

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
    padding(t, stride, pos, theta);
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
        let nh = layout.nh;
        let dh = layout.dh / 2;
        let stride_token = (layout.stride_token / 2) as _;
        let stride_head = (layout.stride_head / 2) as _;
        if self.max_num_threads_block % dh != 0 {
            return Err(locate_error!());
        }
        //                    nh == nh_h * nh_l
        // max_num_threads_block >= nh_l x dh
        let max_nh_l = (self.max_num_threads_block / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;
        Ok(Self::Scheme {
            context: self.context.clone(),
            module: self.module.clone(),
            stream,

            grid: (layout.n as _, nh_h as _),
            block: (nh_l as _, dh as _),
            stride_token,
            stride_head,
        })
    }
}

pub struct Scheme {
    context: Arc<cuda::Context>,
    module: Arc<cuda::ModuleSpore>,
    stream: StreamSpore,

    grid: (c_uint, c_uint),
    block: (c_uint, c_uint),
    stride_token: c_int,
    stride_head: c_int,
}

impl RopeScheme<Gpu> for Scheme {
    fn launch(&self, t: *mut <Gpu as Device>::Byte, pos: *const <Gpu as Device>::Byte, theta: f32) {
        let name = CString::new(NAME).unwrap();
        self.context.apply(|ctx| {
            let stream = self.stream.sprout_ref(ctx);
            let module = self.module.sprout_ref(ctx);
            let kernel = module.get_kernel(&name);
            let params = cuda::params![t, self.stride_token, self.stride_head, pos, theta];
            kernel.launch(self.grid, self.block, params.as_ptr(), 0, Some(&stream));
        });
    }
}
