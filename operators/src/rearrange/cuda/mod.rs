use super::{args::Scheme, Args, Rearrange};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    rank_not_support, shape_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use std::{
    ffi::CString,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    _handle: Arc<Handle>,
    max_warps_block: usize,
    warp_size: usize,
    module: Arc<ModuleBox>,
}

const NAME: &str = "rearrange";
const CODE: &str = include_str!("rearrange.cuh");

impl Rearrange<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        // 提取和检查设备参数
        let device = node.0.device();
        let max_threads_block = device.block_limit().max_threads;
        let warp_size = device.warp_size();
        let cc = device.compute_capability();
        assert_eq!(max_threads_block % warp_size, 0);
        // 生成执行资源
        Self {
            _handle: node.0.clone(),
            max_warps_block: max_threads_block / warp_size,
            warp_size,
            module: node.0.compile_kernel(NAME, cc, format_code),
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        // 完全动态，不需要做任何准备工作
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let scheme = Scheme::new(args)?;
        if scheme.ndim() == 0 {
            let unit = scheme.unit();
            let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let src = unsafe { from_raw_parts(args.src_base, unit) };
            queue_alloc.queue().memcpy_d2d(dst, src);
            return Ok(());
        }

        let scheme = scheme.distribute_unit((0..=5).rev().map(|n| 32 * (1 << n)));
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
            0 => unreachable!(),
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
        let grid = (r, c.div_ceil(warps));
        let block = (c.div_ceil(grid.1), self.warp_size as u32);

        let unit = unit as i32;
        let dst_rs = dst_rs / unit;
        let dst_cs = dst_cs / unit;
        let src_rs = src_rs / unit;
        let src_cs = src_cs / unit;

        let params = cuda::params![
            args.dst_base,
            dst_rs,
            dst_cs,
            args.src_base,
            src_rs,
            src_cs,
            c,
            bytes_thread
        ];
        self.module
            .launch(&name, grid, block, params.as_ptr(), 0, queue_alloc.queue());
        Ok(())
    }
}

fn format_code() -> String {
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
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{ConstPtr, Hardware, MutPtr, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            dst_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        shape: &[usize],
        s_src: &[isize],
        s_dst: &[isize],
        src_base: ConstPtr<H>,
        dst_base: MutPtr<H>,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new(dt, shape, s_dst),
            dst_base,
            src_layout: TensorLayout::new(dt, shape, s_src),
            src_base,
        }
    }

    #[test]
    fn test_compile() {
        use super::NAME;
        use std::ffi::CString;

        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        op.scheme(&dyn_args(ty::F16), 0).unwrap();

        let module = op.module;
        gpu.apply(|ctx| {
            println!(
                "{NAME}\n{}",
                module.load(CString::new(NAME).unwrap(), ctx).info()
            );
        })
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::common_cpu::{Cpu, ThisThread};
        use cuda::memcpy_d2h;
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let dt = ty::U32;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        gpu_op.scheme(&dyn_args(dt), 0).unwrap();

        let nh = 32;
        let seq = 7;
        let dh = 128;
        let mut src = vec![0u32; nh * seq * dh];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, ele);
        let s_dst =
            ArrayLayout::<3>::new_contiguous(&[seq, nh, dh], BigEndian, ele).transpose(&[1, 0]);

        let dst_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let src = rt.from_host(&src);
            let mut dst = rt.malloc::<u8>(src.len());
            gpu_op
                .launch(
                    &args(
                        dt,
                        &[nh, seq, dh],
                        s_src.strides(),
                        s_dst.strides(),
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![0u32; nh * seq * dh];
            memcpy_d2h(&mut host, &dst);
            host
        });

        let mut dst_ref = vec![0u32; seq * nh * dh];
        cpu_op
            .launch(
                &args(
                    dt,
                    &[nh, seq, dh],
                    s_src.strides(),
                    s_dst.strides(),
                    src.as_ptr().cast(),
                    dst_ref.as_mut_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        assert_eq!(dst_ans, dst_ref);
    }
}
