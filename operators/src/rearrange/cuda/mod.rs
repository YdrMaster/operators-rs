use super::{args::Scheme, Args, Rearrange};
use crate::cuda::AsParam;
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    rank_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use itertools::Itertools;
use std::iter::repeat;
use std::{ffi::CString, sync::Arc};
struct Layout {
    r: u32,
    c: u32,
    dst_rs: i32,
    dst_cs: i32,
    src_rs: i32,
    src_cs: i32,
}

const ARRAY_SIZE: usize = 7;

type ArrayType = i32;
struct ArrayStruct([ArrayType; ARRAY_SIZE]);

impl ArrayStruct {
    fn new(element: impl Iterator<Item = ArrayType>, default: ArrayType) -> Option<Self> {
        let mut array = [default; ARRAY_SIZE];
        for (i, v) in element.into_iter().enumerate() {
            if i >= ARRAY_SIZE {
                return None;
            }
            array[i] = v;
        }
        Some(Self(array))
    }
}
//TODO 需要检查正确性
impl AsParam for ArrayStruct {}

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
        // if scheme.ndim() == 0 {
        //     let unit = scheme.unit();
        //     let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
        //     let src = unsafe { from_raw_parts(args.src_base, unit) };
        //     queue_alloc.queue().memcpy_d2d(dst, src);
        //     return Ok(());
        // }

        if scheme.ndim() == 0 {
            let unit = unsafe { BARE_UNIT };
            let len = scheme.unit();

            let name = CString::new(NAME).unwrap();

            // 使用较大的block size来提高并行度
            let block_size = 1024;

            // 计算总元素数
            let total_elements: u32 = (len / unit) as u32;

            let grid_size = (total_elements + block_size - 1) / block_size;

            let params = cuda::params![
                args.dst_base,
                0i32, // rsa
                0i32, // csa
                args.src_base,
                0i32,           // rsb
                0i32,           // csb
                total_elements, // nrows
                1u32,           // ncols
                32u32,          // sub_size_x
                32u32,          // sub_size_y
                unit            // bytes_per_thread
            ];

            self.module.launch(
                &name,
                grid_size as u32,
                block_size as u32,
                params.as_ptr(),
                0,
                queue_alloc.queue(),
            );
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
        let ndim = scheme_update.ndim();

        // // 计算写入的最大连续内存
        // let mut max_dst_contiguous = unit;
        // let mut max_dst_contiguous_index = scheme_update.ndim();
        // for i in (0..scheme_update.ndim()).rev() {
        //     if dst_strides[i] as usize == max_dst_contiguous {
        //         max_dst_contiguous *= shape[i];
        //         max_dst_contiguous_index = i;
        //     } else {
        //         break;
        //     }
        // }

        // // 计算读取的最大连续内存

        // let mut max_src_contiguous = unit;
        // let mut max_src_contiguous_index = scheme_update.ndim();

        //src strides 降序 index
        let src_strides_desc_idx = (0..scheme_update.ndim())
            .zip(src_strides)
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // for i in (0..scheme_update.ndim()).rev() {
        //     if src_strides[src_strides_desc_idx[i]] as usize == max_src_contiguous {
        //         max_src_contiguous *= shape[src_strides_desc_idx[i]];
        //         max_src_contiguous_index = i;
        //     } else {
        //         break;
        //     }
        // }

        //切分维度，分成grid处理的维度和block处理的维度，与dst的维度相对应
        let mut block_dim_choose = repeat(false).take(ndim).collect::<Vec<_>>();
        let mut src_choose_idx = ndim;
        let mut dst_choose_idx = ndim;

        let mut block_elements = 1;
        let mut block_src_elements = 1;
        let mut block_dst_elements = 1;

        //TODO 需要优化
        let block_size = 256;

        loop {
            //优先选择dst
            if block_src_elements > block_dst_elements {
                //选择dst
                let idx = if dst_choose_idx == 0 {
                    break;
                } else {
                    dst_choose_idx - 1
                };

                if block_dim_choose[idx] {
                    dst_choose_idx -= 1;
                    continue;
                }

                if block_elements * shape[idx] <= block_size {
                    block_dim_choose[idx] = true;
                    block_dst_elements *= shape[idx];
                    block_elements *= shape[idx];
                    dst_choose_idx -= 1;
                } else {
                    break;
                }
            } else {
                //选择src
                let idx = if src_choose_idx == 0 {
                    break;
                } else {
                    src_strides_desc_idx[src_choose_idx - 1]
                };

                if block_dim_choose[idx] {
                    dst_choose_idx -= 1;
                    continue;
                }
                if block_elements * shape[src_strides_desc_idx[src_choose_idx - 1]] <= block_size {
                    block_dim_choose[idx] = true;
                    src_choose_idx -= 1;
                    block_src_elements *= shape[idx];

                    block_elements *= shape[idx];
                } else {
                    break;
                }
            }
        }

        println!("block_dim_choose: {:?}", block_dim_choose);

        //TODO 需要支持对单个维度的切分
        if src_choose_idx == ndim || dst_choose_idx == ndim {
            panic!("rearrange not support this scheme");
        }

        //填充block_len，block_stride，grid_len，grid_stride
        macro_rules! fill_array {
            ($input:expr, $default:expr, for_block) => {
                ArrayStruct::new(
                    $input
                        .iter()
                        .zip(block_dim_choose.iter())
                        .filter_map(|(len, is_choose)| {
                            if *is_choose {
                                Some(*len as ArrayType)
                            } else {
                                None
                            }
                        }),
                    $default,
                )
            };

            ($input:expr, $default:expr, for_grid) => {
                ArrayStruct::new(
                    $input
                        .iter()
                        .zip(block_dim_choose.iter())
                        .filter_map(|(len, is_choose)| {
                            if !*is_choose {
                                Some(*len as ArrayType)
                            } else {
                                None
                            }
                        }),
                    $default,
                )
            };
        }

        let block_dim = block_dim_choose.iter().filter(|&&x| x == true).count() as u32;
        let block_len = fill_array!(shape, 1, for_block).unwrap();
        let src_block_stride = fill_array!(src_strides, 0, for_block).unwrap();
        let dst_block_stride = fill_array!(dst_strides, 0, for_block).unwrap();
        let grid_len = fill_array!(shape, 1, for_grid).unwrap();

        let src_grid_stride = fill_array!(src_strides, 0, for_grid).unwrap();
        let dst_grid_stride = fill_array!(dst_strides, 0, for_grid).unwrap();

        //----------------------------------------------------------------------

        let name = CString::new(NAME).unwrap();

        let grid = shape
            .iter()
            .zip(block_dim_choose.iter())
            .filter_map(|(len, is_choose)| {
                if !*is_choose {
                    Some(*len as ArrayType)
                } else {
                    None
                }
            })
            .product::<ArrayType>() as u32;
        let block = block_size as u32;

        let unit = unit as usize;

        let params = cuda::params![
            args.dst_base,
            args.src_base,
            block_dim,
            block_len,        // 各维度的长度
            src_block_stride, // 源tensor在各维度上的步长(bytes)
            dst_block_stride, // 目标tensor在各维度上的步长(bytes)
            grid_len,         // 各维度的长度
            src_grid_stride,  // 源tensor在各维度上的步长(bytes)
            dst_grid_stride,  // 源tensor在各维度上的步长(bytes)
            unit              // bytes_per_thread
        ];

        self.module
            .launch(&name, grid, block, params.as_ptr(), 0, queue_alloc.queue());
        Ok(())
    }
}

fn format_code() -> String {
    format!(
        r#"#define ARRAY_SIZE {ARRAY_SIZE}
        #define ARRAY_TYPE int
        {CODE}

extern "C" __global__ void {NAME}(
    void       *__restrict__ dst,
    void const *__restrict__ src,
    const int block_dim,                   // block维度数量
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> block_len,           // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_block_stride,    // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,    // 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> grid_len,            // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_grid_stride,     // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride,     // 目标tensor在各维度上的步长(bytes)
    unsigned int const unit_size     // 每个元素的字节数
){{
    switch (unit_size) {{
        case  1: rearrange_1<uchar1 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  2: rearrange_1<uchar2 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  4: rearrange_1<float1 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  8: rearrange_1<float2 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case 16: rearrange_1<float4 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case 32: rearrange_1<double4,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
    }}
}}
"#
    )
}

static mut IS_INVERSE: bool = false;
static mut BARE_UNIT: usize = 4;

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
        use crate::rearrange::cuda::format_code;
        use cuda::memcpy_d2h;
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;
        let code = format_code();
        std::fs::write("rearrange.cu", code).unwrap();
        let Some(gpu) = Gpu::init() else {
            return;
        };

        let dt = ty::U64;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        gpu_op.scheme(&dyn_args(dt), 0).unwrap();

        let nh = 16;
        let seq = 3343;
        let dh = 16;
        let mut src = vec![0u64; nh * seq * dh];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, ele);
        let s_dst =
            ArrayLayout::<3>::new_contiguous(&[dh, seq, nh], BigEndian, ele).transpose(&[2, 1, 0]);

        println!("s_src: {:?}", s_src.shape());
        println!("s_dst: {:?}", s_dst.shape());
        println!("s_src strides: {:?}", s_src.strides());

        println!("s_dst strides: {:?}", s_dst.strides());
        let dst_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;

            let src = rt.from_host(&src);
            let mut dst = rt.malloc::<u8>(src.len());

            let start_event = stream.record();

            stream.bench(
                |_, stream| {
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
                            stream,
                        )
                        .unwrap();
                },
                5,
                1,
            );
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

            let mut host = vec![0u64; nh * seq * dh];
            memcpy_d2h(&mut host, &dst);
            host
        });

        let mut dst_ref = vec![0u64; nh * seq * dh];
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
    void *__restrict__ src,
    unsigned int n
){{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {{
        reinterpret_cast<char *>(src)[idx] =  11;
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

        let src_ptr = src.as_mut_ptr();
        let src_len = src.len() as i32;

        let params = cuda::params![src_ptr, src_len];

        module
            .get_kernel(&name)
            .launch(grid as u32, block as u32, params.as_ptr(), 0, Some(queue));
        let _keep_alive = (src_ptr, src_len);
    }

    use std::time::Duration;
    fn time_cost(is_inverse: bool, total_exp: u32, dh_exp: u32) -> Duration {
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
            stream.bench(
                |_, stream| {
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
                            stream,
                        )
                        .unwrap();
                },
                20,
                2,
            )
        })
    }

    fn time_cost_bare(total_exp: u32, dh_exp: u32) -> Duration {
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

        let total_size = 1 << total_exp;
        let unit = 1 << dh_exp;
        use crate::rearrange::cuda::BARE_UNIT;
        unsafe {
            BARE_UNIT = unit;
        }
        let ele = dt.nbytes();
        let s_src = ArrayLayout::<1>::new_contiguous(&[total_size], BigEndian, ele);

        gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let mut src = rt.malloc::<u8>(total_size);
            let mut dst = rt.malloc::<u8>(total_size);
            fill_src(&mut src, ctx, &stream);
            stream.bench(
                |_, stream| {
                    gpu_op
                        .launch(
                            &args(
                                dt,
                                &[total_size],
                                s_src.strides(),
                                s_src.strides(),
                                src.as_ptr().cast(),
                                dst.as_mut_ptr().cast(),
                            ),
                            &mut [],
                            stream,
                        )
                        .unwrap();
                },
                20,
                2,
            )
        })
    }

    #[test]
    fn test_time() {
        for total_exp in [24, 26, 28, 30] {
            println!("\n性能测试结果 (total_exp = {total_exp}):");
            println!(
                "数据规模: {} ({:.2}GB)",
                1u64 << total_exp,
                (1u64 << total_exp) as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!("----------------------------------------");
            println!("dh_exp  dh大小  正向时间          反向时间          直接拷贝时间");
            println!("----------------------------------------");
            for dh_exp in 1..=5 {
                let dh_size = 1 << dh_exp;
                let inverse_time = time_cost(true, total_exp, dh_exp);
                let forward_time = time_cost(false, total_exp, dh_exp);
                let bare_time = time_cost_bare(total_exp, dh_exp);
                println!("{dh_exp:<7} {dh_size:<7} {forward_time:<16?} {inverse_time:<16?} {bare_time:<16?}");
            }
            println!("----------------------------------------");
        }
    }

    #[test]
    fn test_time_one() {
        time_cost(true, 26, 4);
        time_cost(false, 26, 4);
        time_cost_bare(26, 8);
    }
}
