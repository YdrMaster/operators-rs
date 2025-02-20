use super::{args::Scheme, Args, Rearrange};
use crate::cuda::AsParam;
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use itertools::Itertools;
use std::iter::repeat;
use std::{ffi::CString, sync::Arc};

#[derive(Debug)]
struct SplitDim {
    choose_idx: usize,
    num_per_block: usize,
    num_per_grid: usize,
    array_struct_idx_block: ArrayType,
    array_struct_idx_grid: ArrayType,
}

const ARRAY_SIZE: usize = 7;

type ArrayType = i32;
#[derive(Debug)]
struct ArrayStruct<const N: usize>([ArrayType; N]);

impl<const N: usize> ArrayStruct<N> {
    fn new(element: impl Iterator<Item = ArrayType>, default: ArrayType) -> Option<Self> {
        let mut array = [default; N];
        for (i, v) in element.into_iter().enumerate() {
            if i >= N {
                return None;
            }
            array[i] = v;
        }
        Some(Self(array))
    }
}
//TODO 需要检查正确性
impl<const N: usize> AsParam for ArrayStruct<N> {}

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

        //src strides 降序 index
        let src_strides_desc_idx = (0..scheme_update.ndim())
            .zip(src_strides)
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        //分离维度，分成grid处理的维度和block处理的维度，与dst的维度相对应
        let mut block_dim_choose = repeat(false).take(ndim).collect::<Vec<_>>();
        let mut src_choose_idx = ndim;
        let mut dst_choose_idx = ndim;

        let mut block_elements = 1;
        let mut block_src_elements = 1;
        let mut block_dst_elements = 1;

        //TODO 需要优化
        let block_size = 256;
        let mut split_dims = Vec::new(); // 长度最多为2

        while src_choose_idx > 0 && dst_choose_idx > 0 {
            let src_idx = src_strides_desc_idx[src_choose_idx - 1];
            let dst_idx = dst_choose_idx - 1;

            if src_idx == dst_idx {
                let idx = src_idx;
                let len = shape[idx];
                if block_elements * shape[src_idx] <= block_size {
                    //选择维度
                    block_dim_choose[idx] = true;
                    block_elements *= len;
                    block_src_elements *= len;
                    block_dst_elements *= len;
                    src_choose_idx -= 1;
                    dst_choose_idx -= 1;
                } else {
                    //切分维度，并退出
                    let num_per_block = block_size.div_euclid(block_elements);
                    assert!(num_per_block > 0);
                    assert!(len >= num_per_block);
                    if num_per_block > 1 {
                        split_dims.push(SplitDim {
                            choose_idx: idx,
                            num_per_block,
                            num_per_grid: len.div_ceil(num_per_block),
                            array_struct_idx_block: 0,
                            array_struct_idx_grid: 0,
                        });
                    }
                    break;
                }
            } else {
                let src_div_dst = block_src_elements as f64 / block_dst_elements as f64;
                let src_num_per_block =
                    (block_size as f64 / block_elements as f64 / src_div_dst).sqrt();
                let dst_num_per_block = src_num_per_block * src_div_dst;

                let src_current_dim_len = shape[src_idx];
                let dst_current_dim_len = shape[dst_idx];

                if (src_current_dim_len as f64) < src_num_per_block {
                    //选择维度
                    block_dim_choose[src_idx] = true;
                    block_elements *= src_current_dim_len;
                    block_src_elements *= src_current_dim_len;
                    src_choose_idx -= 1;
                } else if (dst_current_dim_len as f64) < dst_num_per_block {
                    //选择维度
                    block_dim_choose[dst_idx] = true;
                    block_elements *= dst_current_dim_len;
                    block_dst_elements *= dst_current_dim_len;
                    dst_choose_idx -= 1;
                } else {
                    //切分维度，并退出
                    let src_num_per_block = src_num_per_block.floor() as usize;
                    let dst_num_per_block = dst_num_per_block.floor() as usize;
                    let src_num_per_grid = src_current_dim_len.div_ceil(src_num_per_block);
                    let dst_num_per_grid = dst_current_dim_len.div_ceil(dst_num_per_block);

                    if src_num_per_block > 1 {
                        split_dims.push(SplitDim {
                            choose_idx: src_idx,
                            num_per_block: src_num_per_block,
                            num_per_grid: src_num_per_grid,
                            array_struct_idx_block: 0,
                            array_struct_idx_grid: 0,
                        });
                    }
                    if dst_num_per_block > 1 {
                        split_dims.push(SplitDim {
                            choose_idx: dst_idx,
                            num_per_block: dst_num_per_block,
                            num_per_grid: dst_num_per_grid,
                            array_struct_idx_block: 0,
                            array_struct_idx_grid: 0,
                        });
                    }
                    break;
                }
            }
        }

        let mut block_dim: ArrayType = 0;

        let mut block_len = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);
        let mut src_block_stride = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);
        let mut dst_block_stride = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);

        let mut grid_len = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);
        let mut src_grid_stride = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);
        let mut dst_grid_stride = Vec::<ArrayType>::with_capacity(ARRAY_SIZE);

        // 处理block，填充block_len，block_stride
        for i in 0..ndim {
            if block_dim_choose[i] {
                block_len.push(shape[i] as ArrayType);
                src_block_stride.push(src_strides[i] as ArrayType);
                dst_block_stride.push(dst_strides[i] as ArrayType);
                block_dim += 1;
            }

            for split_dim in split_dims.iter_mut() {
                if i == split_dim.choose_idx {
                    block_len.push(split_dim.num_per_block as ArrayType);
                    src_block_stride.push(src_strides[i] as ArrayType);
                    dst_block_stride.push(dst_strides[i] as ArrayType);
                    split_dim.array_struct_idx_block = block_dim;
                    block_dim += 1;
                }
            }
        }

        // 处理grid，填充grid_len，grid_stride
        let mut grid_dim = 0;
        for i in 0..ndim {
            let mut is_split = false;
            if !block_dim_choose[i] {
                for split_dim in split_dims.iter_mut() {
                    if i == split_dim.choose_idx {
                        is_split = true;
                        grid_len.push(split_dim.num_per_grid as ArrayType);
                        src_grid_stride
                            .push((src_strides[i] * split_dim.num_per_block as isize) as ArrayType);
                        dst_grid_stride
                            .push((dst_strides[i] * split_dim.num_per_block as isize) as ArrayType);
                        split_dim.array_struct_idx_grid = grid_dim;
                    }
                }
                if !is_split {
                    grid_len.push(shape[i] as ArrayType);
                    src_grid_stride.push(src_strides[i] as ArrayType);
                    dst_grid_stride.push(dst_strides[i] as ArrayType);
                }
                grid_dim += 1;
            }
        }

        println!("split_dims: {:?}", split_dims);
        // cuda 参数准备
        let block_len_total = block_len.iter().product::<ArrayType>();
        let src_block_stride =
            ArrayStruct::<ARRAY_SIZE>::new(src_block_stride.into_iter(), 0).unwrap();
        let dst_block_stride =
            ArrayStruct::<ARRAY_SIZE>::new(dst_block_stride.into_iter(), 0).unwrap();
        let src_grid_stride =
            ArrayStruct::<ARRAY_SIZE>::new(src_grid_stride.into_iter(), 0).unwrap();
        let dst_grid_stride =
            ArrayStruct::<ARRAY_SIZE>::new(dst_grid_stride.into_iter(), 0).unwrap();
        let block_len = ArrayStruct::<ARRAY_SIZE>::new(block_len.into_iter(), 1).unwrap();
        let grid_len = ArrayStruct::<ARRAY_SIZE>::new(grid_len.into_iter(), 1).unwrap();

        let (constrain1, constrain2) = match split_dims.len() {
            0 => (ArrayStruct([0; 4]), ArrayStruct([0; 4])),
            1 => {
                let constrains1 = ArrayStruct([
                    split_dims[0].array_struct_idx_grid,
                    split_dims[0].array_struct_idx_block,
                    split_dims[0].num_per_block as ArrayType,
                    shape[split_dims[0].choose_idx] as ArrayType,
                ]);
                let constrains2 = ArrayStruct([0; 4]);
                (constrains1, constrains2)
            }
            2 => {
                let constrains1 = ArrayStruct([
                    split_dims[0].array_struct_idx_grid,
                    split_dims[0].array_struct_idx_block,
                    split_dims[0].num_per_block as ArrayType,
                    shape[split_dims[0].choose_idx] as ArrayType,
                ]);
                let constrains2 = ArrayStruct([
                    split_dims[1].array_struct_idx_grid,
                    split_dims[1].array_struct_idx_block,
                    split_dims[1].num_per_block as ArrayType,
                    shape[split_dims[1].choose_idx] as ArrayType,
                ]);
                (constrains1, constrains2)
            }
            _ => {
                unreachable!()
            }
        };
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
            block_len_total,
            constrain1,
            constrain2,
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
    const int block_len_total,             // block_len 各元素的乘积
    const ArrayStruct<4, ARRAY_TYPE> constrains1,          // 切分维度的约束条件1
    const ArrayStruct<4, ARRAY_TYPE> constrains2,          // 切分维度的约束条件2
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> block_len,           // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_block_stride,    // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,    // 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> grid_len,            // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_grid_stride,     // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride,     // 目标tensor在各维度上的步长(bytes)
    unsigned int const unit_size     // 每个元素的字节数
){{
    switch (unit_size) {{
        case  1: rearrange_1<uchar1 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  2: rearrange_1<uchar2 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  4: rearrange_1<float1 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case  8: rearrange_1<float2 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case 16: rearrange_1<float4 ,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
        case 32: rearrange_1<double4,ARRAY_SIZE, ARRAY_TYPE>(dst, src, block_dim, block_len_total, constrains1, constrains2, block_len, src_block_stride, dst_block_stride, grid_len, src_grid_stride, dst_grid_stride, unit_size); break;
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

        const N: usize = 5;
        const TRANS_N: usize = 3;
        let shape: [usize; N] = [2232, 3, 7, 9, 4];
        let mut r_shape: [usize; N] = shape.clone();
        r_shape[0..TRANS_N].reverse();

        let trans_param: [usize; TRANS_N] =
            (0..TRANS_N).rev().collect::<Vec<_>>().try_into().unwrap();

        let mut src = vec![0u64; shape.iter().product::<usize>()];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<3>::new_contiguous(&shape, BigEndian, ele);
        let s_dst =
            ArrayLayout::<3>::new_contiguous(&r_shape, BigEndian, ele).transpose(&trans_param);

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
                                &shape,
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
                        &shape,
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

            let mut host = vec![0u64; shape.iter().product::<usize>()];
            memcpy_d2h(&mut host, &dst);
            host
        });

        let mut dst_ref = vec![0u64; shape.iter().product::<usize>()];
        cpu_op
            .launch(
                &args(
                    dt,
                    &shape,
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
