use super::args::{Args, Meta};
use crate::nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox};
use common::{locate_error, Argument, ErrorPosition, QueueOf};
use cuda::Version;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    max_warps_block: usize,
    warp_size: usize,
    scheme: Option<Arc<ModuleBox>>,
}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(handle: &Self::Handle) -> Self {
        let max_threads_block = handle.0.device().block_limit().max_threads;
        let warp_size = handle.0.device().warp_size();
        assert_eq!(max_threads_block % warp_size, 0);
        Self {
            handle: handle.0.clone(),
            max_warps_block: max_threads_block / warp_size,
            warp_size,
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let _ = args.meta()?;
        self.scheme(self.handle.device().compute_capability())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta { dt } = args.meta()?;
        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
        } = args;

        let Some(shape) = Argument::lock(dst_layout.shape()) else {
            return Err(locate_error!("dst_layout.shape is dynamic"));
        };
        let Some(shape_) = Argument::lock(src_layout.shape()) else {
            return Err(locate_error!("src_layout.shape is dynamic"));
        };
        if shape != shape_ {
            return Err(locate_error!());
        }
        let Some(dst_strides) = Argument::lock(dst_layout.strides()) else {
            return Err(locate_error!("dst_layout.strides is dynamic"));
        };
        let Some(src_strides) = Argument::lock(src_layout.strides()) else {
            return Err(locate_error!("src_layout.strides is dynamic"));
        };

        let [r @ .., c, z] = shape else {
            unreachable!()
        };
        let [dst_rs @ .., dst_cs, dst_zs] = dst_strides else {
            unreachable!()
        };
        let [src_rs @ .., src_cs, src_zs] = src_strides else {
            unreachable!()
        };
        assert_eq!(dst_rs.len(), r.len());
        assert_eq!(src_rs.len(), r.len());

        let unit = dt.nbytes() as isize;
        if *dst_zs != unit || *src_zs != unit {
            return Err(locate_error!());
        }

        let Some(m) = self.scheme.as_ref() else {
            return Err(locate_error!("scheme is not set"));
        };

        let r = match r {
            [] => 1,
            &[r] => r,
            r => {
                for (i, r) in r.iter().enumerate().skip(1) {
                    let r = *r as isize;
                    if r * dst_rs[i] != dst_rs[i - 1] || r * src_rs[i] != src_rs[i - 1] {
                        return Err(locate_error!("目前要求前序维度必须连续"));
                    }
                }
                r.iter().product()
            }
        };

        let z = z * unit as usize;

        if z % self.warp_size != 0 {
            return Err(locate_error!());
        }
        let bytes_thread = z / self.warp_size;
        if bytes_thread > 32 || !bytes_thread.is_power_of_two() {
            return Err(locate_error!());
        }

        let grid = (r, (c + self.max_warps_block - 1) / self.max_warps_block);
        let block = ((c + grid.1 - 1) / grid.1, self.warp_size);

        let grid = (grid.0 as u32, grid.1 as u32);
        let block = (block.0 as u32, block.1 as u32);
        let dst_rs = dst_rs.last().copied().unwrap_or(0) as i32 / z as i32;
        let dst_cs = *dst_cs as i32 / z as i32;
        let src_rs = src_rs.last().copied().unwrap_or(0) as i32 / z as i32;
        let src_cs = *src_cs as i32 / z as i32;
        let c = *c as u32;
        let bytes_thread = bytes_thread as u32;

        let name = CString::new(NAME).unwrap();
        let params = cuda::params![
            dst_base,
            dst_rs,
            dst_cs,
            src_base,
            src_rs,
            src_cs,
            c,
            bytes_thread
        ];
        m.launch(&name, grid, block, params.as_ptr(), 0, queue);

        Ok(())
    }
}

const NAME: &str = "reform";
const CODE: &str = include_str!("reform.cuh");
impl Operator {
    fn scheme(&mut self, cc: Version) -> Result<(), ErrorPosition> {
        let module = self
            .handle
            .compile(NAME, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {NAME}(
    void       *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
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
                )
            })
            .map_err(|(e, log)| locate_error!(format!("Failed to compile {NAME}: {e:?}\n{log}")))?;
        self.scheme = Some(module);
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    <Operator as common::Operator>::scheme(
        &mut op,
        &Args {
            dst_layout: TensorLayout::new(F16, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new(F16, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
        },
    )
    .unwrap();
    let module = op.scheme.as_ref().unwrap();
    handle.apply(|ctx| {
        println!(
            "{NAME}\n{}",
            module.load(CString::new(NAME).unwrap(), ctx).info()
        );
    })
}
