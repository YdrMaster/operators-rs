use super::{layout::RmsNormSchemeLayout, RmsNormScheme, RmsNormTensorLayout};
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
    pub num_items_reduce: usize,
    pub num_threads_warp: usize,
    pub max_num_threads_block: usize,
    pub compute_capability: ComputeCapability,
}

impl Config {
    pub fn config_for(
        dev: &cuda::Device,
        data_layout: DataLayout,
        num_items_reduce: usize,
    ) -> Self {
        Self {
            data_layout,
            num_items_reduce,
            num_threads_warp: 32,
            max_num_threads_block: dev.max_block_dims().0,
            compute_capability: dev.compute_capability(),
        }
    }
}

pub struct Operator {
    ptx: Ptx,
    name: CString,
    ty: KernelType,
}

impl crate::Operator<Gpu> for Operator {
    type Config = Config;
    type ConfigError = ErrorPosition;

    fn config(
        Self::Config {
            data_layout,
            num_items_reduce,
            num_threads_warp,
            max_num_threads_block,
            compute_capability,
        }: Self::Config,
    ) -> Result<Self, Self::ConfigError> {
        const RMS_NORMALIZATION: &str = include_str!("rms_norm.cuh");

        if data_layout != F16 {
            return Err(locate_error!());
        }
        if num_items_reduce <= max_num_threads_block {
            let name = format!("rms_norm_padding_f16_{num_items_reduce}");
            let code = format!(
                r#"{RMS_NORMALIZATION}

extern "C" __global__ void {name}(
    half *__restrict__ y,
    int  const stride_y,
    half const *__restrict__ x,
    int  const stride_x,
    half const *__restrict__ w,
    float epsilon
){{
    padding<{num_items_reduce}>
    (y, stride_y, x, stride_x, w, epsilon);
}}"#
            );

            let (ptx, log) = Ptx::compile(code, compute_capability);
            let ptx =
                ptx.map_err(|e| locate_error!(format!("Failed to compile {name}: {e:?}\n{log}")))?;
            if !log.is_empty() {
                warn!("{log}");
            }

            Ok(Self {
                ptx,
                name: CString::new(name).unwrap(),
                ty: KernelType::Padding { num_items_reduce },
            })
        } else {
            if max_num_threads_block % num_threads_warp != 0 {
                return Err(locate_error!());
            }
            if num_items_reduce % num_threads_warp != 0 {
                return Err(locate_error!());
            }
            let max_num_warp_block = max_num_threads_block / num_threads_warp;
            // num_warp_block in [1, max_num_warp_block]
            // num_threads_warp
            // num_items_thread in [1, 2, 4, 8] // 8 = 128bit / sizeof(half)
            // TODO 也许还能分得更好
            let to_divid = num_items_reduce / num_threads_warp;
            let num_warps_block = max_num_warp_block;
            let num_threads_block = num_threads_warp * num_warps_block;
            let num_items_thread = (to_divid + num_warps_block - 1) / num_warps_block;

            let name = format!("rms_norm_folding_f16_{num_items_reduce}");
            let code = format!(
                r#"{RMS_NORMALIZATION}

extern "C" __global__ void {name}(
    half *__restrict__ y,
    int  const stride_y,
    half const *__restrict__ x,
    int  const stride_x,
    half const *__restrict__ w,
    float epsilon
){{
    folding<{num_threads_block}, {num_items_thread}>
    (y, stride_y, x, stride_x, w, epsilon, {num_items_reduce});
}}"#
            );

            let (ptx, log) = Ptx::compile(code, compute_capability);
            let ptx =
                ptx.map_err(|e| locate_error!(format!("Failed to compile {name}: {e:?}\n{log}")))?;
            if !log.is_empty() {
                warn!("{log}");
            }

            Ok(Self {
                ptx,
                name: CString::new(name).unwrap(),
                ty: KernelType::Folding {
                    num_threads_block: num_warps_block * num_threads_warp,
                    num_items_reduce,
                },
            })
        }
    }

    type Kernel = Kernel;
    type LoadError = ();

    fn load(&self, ctx: &<Gpu as Device>::Context) -> Result<Self::Kernel, Self::LoadError> {
        Ok(Kernel {
            context: ctx.clone(),
            module: Arc::new(ctx.apply(|ctx| ctx.load(&self.ptx).sporulate())),
            name: self.name.clone(),
            ty: self.ty.clone(),
        })
    }
}

pub struct Kernel {
    context: Arc<cuda::Context>,
    module: Arc<cuda::ModuleSpore>,
    name: CString,
    ty: KernelType,
}

#[derive(Clone, Debug)]
enum KernelType {
    Padding {
        num_items_reduce: usize,
    },
    Folding {
        num_threads_block: usize,
        num_items_reduce: usize,
    },
}

impl crate::Kernel<Gpu> for Kernel {
    type Scheme = Scheme;
    type Config = (RmsNormTensorLayout, StreamSpore);
    type SchemeError = ErrorPosition;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError> {
        let (layout, stream) = config;
        let layout = RmsNormSchemeLayout::new(F16, layout)?;
        match self.ty {
            KernelType::Padding { num_items_reduce } => {
                if layout.d != num_items_reduce {
                    return Err(locate_error!());
                }
                Ok(Self::Scheme {
                    context: self.context.clone(),
                    module: self.module.clone(),
                    name: self.name.clone(),
                    stream,
                    num_blocks_grid: layout.n as _,
                    num_threads_block: layout.d as _,
                    stride_y: layout.stride_y as _,
                    stride_x: layout.stride_x as _,
                })
            }
            KernelType::Folding {
                num_threads_block,
                num_items_reduce,
            } => {
                if layout.d != num_items_reduce {
                    return Err(locate_error!());
                }
                Ok(Self::Scheme {
                    context: self.context.clone(),
                    module: self.module.clone(),
                    name: self.name.clone(),
                    stream,
                    num_blocks_grid: layout.n as _,
                    num_threads_block: num_threads_block as _,
                    stride_y: layout.stride_y as _,
                    stride_x: layout.stride_x as _,
                })
            }
        }
    }
}

pub struct Scheme {
    context: Arc<cuda::Context>,
    module: Arc<cuda::ModuleSpore>,
    name: CString,
    stream: StreamSpore,

    num_blocks_grid: c_uint,
    num_threads_block: c_uint,
    stride_y: c_int,
    stride_x: c_int,
}

impl RmsNormScheme<Gpu> for Scheme {
    fn launch(
        &self,
        y: *mut <Gpu as Device>::Byte,
        x: *const <Gpu as Device>::Byte,
        w: *const <Gpu as Device>::Byte,
        epsilon: f32,
    ) {
        self.context.apply(|ctx| {
            let stream = unsafe { self.stream.sprout(ctx) };
            let module = unsafe { self.module.sprout(ctx) };
            let kernel = module.get_kernel(&self.name);
            let params = cuda::params![y, self.stride_y, x, self.stride_x, w, epsilon];
            kernel.launch(
                self.num_blocks_grid,
                self.num_threads_block,
                params.as_ptr(),
                0,
                Some(&stream),
            );
        });
    }
}
