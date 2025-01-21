use super::{args::Scheme, Args, Rearrange};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    rank_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use itertools::Itertools;
use std::{
    ffi::CString,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};
struct Layout {
    r: u32,
    c: u32,
    dst_rs: i32,
    dst_cs: i32,
    src_rs: i32,
    src_cs: i32,
}

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

        //----------------------------------------------------------------------
        // 发现读取的最大连续内存和写入的最大连续内存

        // 发现最大的1 thread 处理的数据量
        let scheme_update = scheme.distribute_unit((0..=5).rev().map(|n| (1 << n)));

        let src_strides = scheme_update.src_strides();
        let dst_strides = scheme_update.dst_strides();
        let shape = scheme_update.shape().collect::<Vec<_>>();
        let unit = scheme_update.unit();

        // 计算写入的最大连续内存
        let mut max_dst_contiguous = unit;
        let mut max_dst_contiguous_index = scheme_update.ndim();
        for i in (0..scheme_update.ndim()).rev() {
            if dst_strides[i] as usize == max_dst_contiguous {
                max_dst_contiguous *= shape[i];
                max_dst_contiguous_index = i;
            } else {
                break;
            }
        }

        // 计算读取的最大连续内存

        let mut max_src_contiguous = unit;
        let mut max_src_contiguous_index = scheme_update.ndim();

        //src strides 降序 index
        let src_strides_desc_idx = (0..scheme_update.ndim())
            .zip(src_strides)
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        for i in (0..scheme_update.ndim()).rev() {
            if src_strides[src_strides_desc_idx[i]] as usize == max_src_contiguous {
                max_src_contiguous *= shape[src_strides_desc_idx[i]];
                max_src_contiguous_index = i;
            } else {
                break;
            }
        }

        // 检查源数据和目标数据的连续性
        let src_continuous = if max_src_contiguous_index == scheme.ndim() {
            false
        } else {
            shape[*src_strides_desc_idx.last().unwrap()] >= self.warp_size
        };
        let dst_continuous = if max_dst_contiguous_index == scheme.ndim() {
            false
        } else {
            *shape.last().unwrap() >= self.warp_size
        };

        // println!("src_continuous: {src_continuous}, dst_continuous: {dst_continuous}");

        // 决定是否使用共享内存优化
        let _use_shared_memory = src_continuous || dst_continuous;

        let use_shared_memory = true;
        //----------------------------------------------------------------------

        let layout = match scheme_update.ndim() {
            0 => unreachable!(),
            1 => {
                let &[dst_cs] = scheme_update.dst_strides() else {
                    unreachable!()
                };
                let &[src_cs] = scheme_update.src_strides() else {
                    unreachable!()
                };
                Layout {
                    r: 1,
                    c: scheme_update.shape().next().unwrap() as _,
                    dst_rs: 0,
                    dst_cs: dst_cs as _,
                    src_rs: 0,
                    src_cs: src_cs as _,
                }
            }
            2 => {
                let mut shape = scheme_update.shape();
                let r = shape.next().unwrap();
                let c = shape.next().unwrap();
                let &[dst_rs, dst_cs] = scheme_update.dst_strides() else {
                    unreachable!()
                };
                let &[src_rs, src_cs] = scheme_update.src_strides() else {
                    unreachable!()
                };

                unsafe {
                    if IS_INVERSE {
                        Layout {
                            r: c as _,
                            c: r as _,
                            dst_rs: dst_cs as _,
                            dst_cs: dst_rs as _,
                            src_rs: src_cs as _,
                            src_cs: src_rs as _,
                        }
                    } else {
                        Layout {
                            r: r as _,
                            c: c as _,
                            dst_rs: dst_rs as _,
                            dst_cs: dst_cs as _,
                            src_rs: src_rs as _,
                            src_cs: src_cs as _,
                        }
                    }
                }
            }
            _ => Err(rank_not_support("rearrange not support ndim > 2 on NV GPU"))?,
        };

        // println!(
        //     "layout: r={}, c={}, dst_rs={}, dst_cs={}, src_rs={}, src_cs={}",
        //     layout.r, layout.c, layout.dst_rs, layout.dst_cs, layout.src_rs, layout.src_cs
        // );

        let name = CString::new(NAME).unwrap();

        let warps = self.max_warps_block as u32;
        let grid = (
            layout.c.div_ceil(self.warp_size as u32),
            layout.r.div_ceil(self.warp_size as u32),
        );
        let block = if use_shared_memory {
            // 使用32x32的block大小
            (32, 32)
        } else {
            (self.warp_size as u32, warps)
        };

        let unit = unit as i32;
        let dst_rs = layout.dst_rs / unit;
        let dst_cs = layout.dst_cs / unit;
        let src_rs = layout.src_rs / unit;
        let src_cs = layout.src_cs / unit;

        let params = cuda::params![
            args.dst_base,
            dst_rs,
            dst_cs,
            args.src_base,
            src_rs,
            src_cs,
            layout.r,
            layout.c,
            32u32,         // sub_size_x
            32u32,         // sub_size_y
            (unit as u32)  // bytes_per_thread
        ];

        let shared_memory_size = if use_shared_memory {
            // 计算共享内存大小：32x32 的块大小 * 每个元素的字节数
            32 * 32 * (unit as usize)
        } else {
            0
        };

        self.module.launch(
            &name,
            grid,
            block,
            params.as_ptr(),
            shared_memory_size,
            queue_alloc.queue(),
        );
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
    unsigned int const nrows,
    unsigned int const ncols,
    unsigned int const sub_size_x,
    unsigned int const sub_size_y,
    unsigned int const bytes_per_thread
){{
    // 使用共享内存版本
    switch (bytes_per_thread) {{  // 使用实际的 bytes_per_thread
        case  1: rearrange_shared<uchar1 >(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
        case  2: rearrange_shared<uchar2 >(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
        case  4: rearrange_shared<float1 >(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
        case  8: rearrange_shared<float2 >(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
        case 16: rearrange_shared<float4 >(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
        case 32: rearrange_shared<double4>(dst, rsa, csa, src, rsb, csb, nrows, ncols, sub_size_x, sub_size_y); break;
    }}
}}
"#
    )
}

static mut IS_INVERSE: bool = false;

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{ConstPtr, Hardware, MutPtr, Operator as _, TensorLayout};
    use cuda::{DevMem, Ptx};
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
        op.scheme(&dyn_args(ty::U8), 0).unwrap();

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

        let nh = 4342;
        let seq = 143;
        let dh = 8;
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

            let start_event = stream.record();
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
            let end_event = stream.record();
            end_event.synchronize();
            let time = end_event.elapse_from(&start_event);
            println!("time: {time:?}");
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

    use crate::cuda::CurrentCtx;
    use crate::cuda::Stream;

    use std::ffi::CString;
    fn fill_src_code() -> String {
        format!(
            r#"

extern "C" __global__ void fill_src(
    unsigned char *src,
    unsigned int n
){{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        ((unsigned char*)src)[idx] = threadIdx.x;
    }}
}}
"#
        )
    }
    fn fill_src(src: &mut DevMem, ctx: &CurrentCtx, queue: &Stream) {
        let (ptx, _) = Ptx::compile(fill_src_code(), ctx.dev().compute_capability());
        let module = ctx.load(&ptx.unwrap());
        let name = CString::new("fill_src").unwrap();

        let block_size = 256; // 使用较小的 block size
        let total_threads = src.len();

        let grid_size = (total_threads + block_size - 1) / block_size;

        let block = block_size;
        let grid = grid_size;

        let params = cuda::params![src.as_mut_ptr(), src.len() as u32];
        module
            .get_kernel(&name)
            .launch(grid as u32, block as u32, params.as_ptr(), 0, Some(queue));
    }

    use std::time;
    fn time_cost(is_inverse: bool, total_exp: u32, dh_exp: u32) -> time::Duration {
        use super::super::common_cpu::Operator as RefOp;
        use crate::common_cpu::Cpu;

        use ndarray_layout::{ArrayLayout, Endian::BigEndian};

        let Some(gpu) = Gpu::init() else {
            panic!("init gpu failed");
        };

        let dt = ty::U8;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        gpu_op.scheme(&dyn_args(dt), 0).unwrap();

        let nh = 1 << ((total_exp + 1) / 2 - (dh_exp + 1) / 2);
        let seq = 1 << (total_exp / 2 - dh_exp / 2);
        let dh = 1 << dh_exp;
        // println!("nh: {nh}, seq: {seq}, dh: {dh}");

        let ele = dt.nbytes();

        let s_src = ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, ele);
        let s_dst =
            ArrayLayout::<3>::new_contiguous(&[seq, nh, dh], BigEndian, ele).transpose(&[1, 0]);

        use super::IS_INVERSE;
        unsafe {
            IS_INVERSE = is_inverse;
        }
        gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let mut src = rt.malloc::<u8>(nh * seq * dh);
            let mut dst = rt.malloc::<u8>(nh * seq * dh);
            fill_src(&mut src, ctx, &stream);

            let mut total_time = time::Duration::ZERO;

            let count = 10;
            for _ in 0..count {
                let start_event = stream.record();
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
                let end_event = stream.record();
                end_event.synchronize();
                let time = end_event.elapse_from(&start_event);
                // println!("time: {time:?}");
                total_time += time;
            }
            total_time / count
        })
    }

    #[test]
    fn test_time() {
        let total_exps = [24, 26, 28, 30, 32];

        for &total_exp in &total_exps {
            // 收集测试数据
            let mut results = Vec::new();

            println!("\n性能测试结果 (total_exp = {total_exp}):");
            println!(
                "数据规模: {} ({:.2}GB)",
                1u64 << total_exp,
                (1u64 << total_exp) as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!("----------------------------------------");
            println!("dh_exp  dh大小  正向时间          反向时间");
            println!("----------------------------------------");

            for dh_exp in 1..=5 {
                let dh_size = 1 << dh_exp;
                let forward_time = time_cost(false, total_exp, dh_exp);
                let inverse_time = time_cost(true, total_exp, dh_exp);
                results.push((dh_exp, dh_size, forward_time, inverse_time));

                println!(
                    "{:<7} {:<7} {:<16?} {:<16?}",
                    dh_exp, dh_size, forward_time, inverse_time
                );
            }

            println!("----------------------------------------");

            // 计算和打印平均时间
            let avg_forward = results
                .iter()
                .map(|(_, _, f, _)| f.as_nanos())
                .sum::<u128>()
                / results.len() as u128;
            let avg_inverse = results
                .iter()
                .map(|(_, _, _, i)| i.as_nanos())
                .sum::<u128>()
                / results.len() as u128;
            println!(
                "平均时间: 正向={:?}ns, 反向={:?}ns",
                avg_forward, avg_inverse
            );
            println!("========================================\n");
        }
    }
}
