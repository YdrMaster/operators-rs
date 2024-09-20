use super::{args::Scheme, Args, Rearrange};
use crate::nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox};
use common::{
    rank_not_support, scheme_not_set, shape_not_support, LaunchError, QueueOf, SchemeError,
};
use dev_mempool::cuda::Version;
use std::{
    ffi::CString,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    handle: Arc<Handle>,
    max_warps_block: usize,
    warp_size: usize,
    scheme: Option<Arc<ModuleBox>>,
}

impl Rearrange<Gpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;

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

    fn scheme(&mut self, _args: &Self::Args) -> Result<(), SchemeError> {
        self.scheme(self.handle.device().compute_capability())
    }

    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
        let scheme = Scheme::new(args)?.distribute_unit((0..=5).rev().map(|n| 32 * (1 << n)));
        let unit = scheme.unit();

        struct Layout {
            r: u32,
            c: u32,
            dst_rs: i32,
            dst_cs: i32,
            src_rs: i32,
            src_cs: i32,
        }

        let Layout {
            r,
            c,
            dst_rs,
            dst_cs,
            src_rs,
            src_cs,
        } = match scheme.ndim() {
            0 => {
                let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
                let src = unsafe { from_raw_parts(args.src_base, unit) };
                queue.memcpy_d2d(dst, src);
                return Ok(());
            }
            1 => {
                let &[dst_cs] = scheme.dst_strides() else {
                    unreachable!()
                };
                let &[src_cs] = scheme.src_strides() else {
                    unreachable!()
                };
                Layout {
                    r: 1,
                    c: scheme.shape().next().unwrap() as _,
                    dst_rs: 0,
                    dst_cs: dst_cs as _,
                    src_rs: 0,
                    src_cs: src_cs as _,
                }
            }
            2 => {
                let mut shape = scheme.shape();
                let r = shape.next().unwrap();
                let c = shape.next().unwrap();
                let &[dst_rs, dst_cs] = scheme.dst_strides() else {
                    unreachable!()
                };
                let &[src_rs, src_cs] = scheme.src_strides() else {
                    unreachable!()
                };
                Layout {
                    r: r as _,
                    c: c as _,
                    dst_rs: dst_rs as _,
                    dst_cs: dst_cs as _,
                    src_rs: src_rs as _,
                    src_cs: src_cs as _,
                }
            }
            _ => Err(rank_not_support("rearrange not support ndim > 2 on NV GPU"))?,
        };

        let name = CString::new(NAME).unwrap();
        let Some(m) = self.scheme.as_ref() else {
            return Err(scheme_not_set(""));
        };
        if unit % self.warp_size != 0 {
            Err(shape_not_support(format!(
                "memory region {unit} is not align to warp size, which is not supported yet on NV GPU",
            )))?;
        }
        let bytes_thread = (unit / self.warp_size) as u32;
        if bytes_thread > 32 || !bytes_thread.is_power_of_two() {
            Err(shape_not_support(format!(
                "bytes per thread {bytes_thread} is not supported yet on NV GPU"
            )))?;
        }

        let warps = self.max_warps_block as u32;
        let grid = (r, (c + warps - 1) / warps);
        let block = ((c + grid.1 - 1) / grid.1, self.warp_size as u32);

        let unit = unit as i32;
        let dst_rs = dst_rs / unit;
        let dst_cs = dst_cs / unit;
        let src_rs = src_rs / unit;
        let src_cs = src_cs / unit;

        let params = dev_mempool::cuda::params![
            args.dst_base,
            dst_rs,
            dst_cs,
            args.src_base,
            src_rs,
            src_cs,
            c,
            bytes_thread
        ];
        m.launch(&name, grid, block, params.as_ptr(), 0, queue);
        Ok(())
    }
}

const NAME: &str = "rearrange";
const CODE: &str = include_str!("rearrange.cuh");
impl Operator {
    fn scheme(&mut self, cc: Version) -> Result<(), SchemeError> {
        self.scheme = Some(self.handle.compile_kernel(NAME, cc, || {
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
        case  1: rearrange<uchar1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  2: rearrange<uchar2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  4: rearrange<float1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  8: rearrange<float2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 16: rearrange<float4 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 32: rearrange<double4>(dst, rsa, csa, src, rsb, csb, ncols); break;
    }}
}}
"#
            )
        }));
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use dev_mempool::cuda;
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};

    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let dev = cuda::Device::new(0);
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    <Operator as common::Operator>::scheme(
        &mut op,
        &Args {
            dst_layout: TensorLayout::new_dyn(F16, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(F16, &[dyn_(); 2], &[dyn_(); 2]),
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
