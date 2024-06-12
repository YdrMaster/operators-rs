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

#[derive(Clone, Debug)]
pub struct Operator {
    global: __global__,
    name: CString,
    ty: KernelType,
}

impl common::Operator<Gpu> for Operator {
    type Config = Config;
    type Error = ErrorPosition;

    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        let &Self::Config {
            data_layout,
            num_items_reduce,
            num_threads_warp,
            max_num_threads_block,
            compute_capability,
        } = config;

        const CODE: &str = include_str!("rms_norm.cuh");

        if data_layout != F16 {
            return Err(locate_error!());
        }
        if num_items_reduce <= max_num_threads_block {
            let name = format!("rms_norm_padding_f16_{num_items_reduce}");
            let code = format!(
                r#"{CODE}

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
                global: __global__::load(&ptx),
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
                r#"{CODE}

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
                global: __global__::load(&ptx),
                name: CString::new(name).unwrap(),
                ty: KernelType::Folding {
                    num_threads_block: num_warps_block * num_threads_warp,
                    num_items_reduce,
                },
            })
        }
    }
}

pub struct Scheme {
    global: __global__,
    name: CString,

    num_blocks_grid: c_uint,
    num_threads_block: c_uint,
    stride_y: c_int,
    stride_x: c_int,
    offset_y: usize,
    offset_x: usize,
    offset_w: usize,
}

impl common::Scheme<Gpu, Operator> for Scheme {
    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        let SchemeLayout {
            n,
            d,
            stride_y,
            stride_x,
            offset_y,
            offset_x,
            offset_w,
        } = SchemeLayout::new(F16, layout)?;

        match op.ty {
            KernelType::Padding { num_items_reduce } => {
                if d != num_items_reduce {
                    return Err(locate_error!());
                }
                Ok(Self {
                    global: op.global,
                    name: op.name.clone(),

                    num_blocks_grid: n as _,
                    num_threads_block: d as _,
                    stride_y: stride_y as _,
                    stride_x: stride_x as _,
                    offset_y,
                    offset_x,
                    offset_w,
                })
            }
            KernelType::Folding {
                num_threads_block,
                num_items_reduce,
            } => {
                if d != num_items_reduce {
                    return Err(locate_error!());
                }
                Ok(Self {
                    global: op.global,
                    name: op.name.clone(),

                    num_blocks_grid: n as _,
                    num_threads_block: num_threads_block as _,
                    stride_y: stride_y as _,
                    stride_x: stride_x as _,
                    offset_y,
                    offset_x,
                    offset_w,
                })
            }
        }
    }

    type Params<'ctx> = Params<Gpu>;
    fn launch(&self, params: &Self::Params<'_>, queue: &QueueOf<Gpu>) {
        let &(y, x, w, epsilon) = params;
        let y = unsafe { y.add(self.offset_y) };
        let x = unsafe { x.add(self.offset_x) };
        let w = unsafe { w.add(self.offset_w) };
        let params = cuda::params![y, self.stride_y, x, self.stride_x, w, epsilon];
        self.global.launch(
            &self.name,
            self.num_blocks_grid,
            self.num_threads_block,
            params.as_ptr(),
            0,
            queue,
        );
    }
}
