use super::{args::Scheme as ArgsScheme, Args, Rearrange};
use crate::cuda::AsParam;

use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    shape_not_support, ByteOf, LaunchError, QueueAlloc, SchemeDiversity, SchemeError,
};

use itertools::Itertools;
use lru::LruCache;
use std::iter::repeat;
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    unit_size: usize,
    constrain_num: usize,
}

#[derive(Clone)]
struct Scheme {
    module: Arc<ModuleBox>,
    name: CString,
}

impl Scheme {
    pub fn new(key: SchemeKey, handle: &Arc<Handle>) -> Result<Self, SchemeError> {
        let name = kernel_name(key.unit_size, key.constrain_num);
        let cc = handle.device().compute_capability();
        let code = format_code(key.unit_size, key.constrain_num);
        std::fs::write("rearrange.cu", code).unwrap();
        Ok(Self {
            module: handle
                .compile_kernel(&name, cc, || format_code(key.unit_size, key.constrain_num)),
            name: CString::new(name).unwrap(),
        })
    }
}

/// Maximum number of dimensions supported in the array
const ARRAY_SIZE: usize = 5;
/// Type used for array indices and strides
type ArrayType = i32;

#[derive(Debug)]
struct SplitDim {
    choose_idx: usize,
    num_per_block: usize,
    num_per_grid: usize,
    array_struct_idx_block: ArrayType,
    array_struct_idx_grid: ArrayType,
    dim_len: usize,
}

#[derive(Debug)]
struct ArrayStruct<const N: usize>([ArrayType; N]);

impl<const N: usize> ArrayStruct<N> {
    fn new(
        element: impl Iterator<Item = ArrayType>,
        default: ArrayType,
    ) -> Result<Self, SchemeError> {
        let mut array = [default; N];
        for (i, v) in element.into_iter().enumerate() {
            if i >= N {
                return Err(shape_not_support(
                    "ArrayStruct::new: too many elements".to_string(),
                ));
            }
            array[i] = v;
        }
        Ok(Self(array))
    }
}

impl<const N: usize> AsParam for ArrayStruct<N> {}

//TODO 需要使用max_warps_block和warp_size来进行计算
pub struct Operator {
    handle: Arc<Handle>,
    #[allow(unused)]
    max_warps_block: usize,
    #[allow(unused)]
    warp_size: usize,
    schemes: Mutex<LruCache<SchemeKey, Scheme>>,
}

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
        assert_eq!(max_threads_block % warp_size, 0);
        // 生成执行资源
        Self {
            handle: node.0.clone(),
            max_warps_block: max_threads_block / warp_size,
            warp_size,
            schemes: node.0.scheme_cache(SchemeDiversity::Low),
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        //TODO 目前不支持dyn_args
        // println!("scheme");
        // let scheme_update = ArgsScheme::new(args)?;
        // println!("scheme_update: {:?}", scheme_update);
        // // 发现最大的1 thread 处理的数据量
        // let scheme_update = scheme_update.distribute_unit((0..=5).rev().map(|n| (1 << n)));

        // let unit_size = scheme_update.unit();

        // // 编译所有可能的约束数量kernel
        // for constrain_num in 0..=2 {
        //     let key = SchemeKey {
        //         unit_size,
        //         constrain_num,
        //     };

        //     self.schemes
        //         .lock()
        //         .unwrap()
        //         .try_get_or_insert(key, || Scheme::new(key, &self.handle))?;
        // }
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
        let scheme_update = ArgsScheme::new(args)?;

        // 发现最大的1 thread 处理的数据量
        let scheme_update = scheme_update.distribute_unit((0..=5).rev().map(|n| (1 << n)));

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

        //TODO 需要优化
        let max_block_size = 256;
        let mut split_dims = Vec::new(); // 长度最多为2

        {
            let mut src_choose_idx = ndim;
            let mut dst_choose_idx = ndim;

            let mut block_elements = 1;
            let mut block_src_elements = 1;
            let mut block_dst_elements = 1;

            while src_choose_idx > 0 && dst_choose_idx > 0 {
                let src_idx = src_strides_desc_idx[src_choose_idx - 1];
                let dst_idx = dst_choose_idx - 1;

                if src_idx == dst_idx {
                    let idx = src_idx;
                    let len = shape[idx];
                    if block_elements * shape[src_idx] <= max_block_size {
                        //选择维度
                        block_dim_choose[idx] = true;
                        block_elements *= len;
                        block_src_elements *= len;
                        block_dst_elements *= len;
                        src_choose_idx -= 1;
                        dst_choose_idx -= 1;
                    } else {
                        //切分维度，并退出
                        let num_per_block = max_block_size.div_euclid(block_elements);
                        assert!(num_per_block > 0);
                        assert!(len >= num_per_block);
                        if num_per_block > 1 {
                            split_dims.push(SplitDim {
                                choose_idx: idx,
                                num_per_block,
                                num_per_grid: len.div_ceil(num_per_block),
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: len,
                            });
                        }
                        break;
                    }
                } else {
                    let src_div_dst = block_src_elements as f64 / block_dst_elements as f64;
                    let src_num_per_block =
                        (max_block_size as f64 / block_elements as f64 / src_div_dst).sqrt();
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

                        if src_num_per_block == 1 {
                        } else if src_num_per_grid == 1 {
                            block_dim_choose[src_idx] = true;
                        } else {
                            split_dims.push(SplitDim {
                                choose_idx: src_idx,
                                num_per_block: src_num_per_block,
                                num_per_grid: src_num_per_grid,
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: src_current_dim_len,
                            });
                        }

                        if dst_num_per_block == 1 {
                        } else if dst_num_per_grid == 1 {
                            block_dim_choose[dst_idx] = true;
                        } else {
                            split_dims.push(SplitDim {
                                choose_idx: dst_idx,
                                num_per_block: dst_num_per_block,
                                num_per_grid: dst_num_per_grid,
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: dst_current_dim_len,
                            });
                        }
                        break;
                    }
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
        let mut grid_dim = 0 as u32;
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
                        split_dim.array_struct_idx_grid = grid_dim as ArrayType;
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

        let constrain_num = split_dims.len();

        let key = SchemeKey {
            unit_size: unit,
            constrain_num,
        };

        let mut schemes = self.schemes.lock().unwrap();
        let scheme = schemes.try_get_or_insert(key, || Scheme::new(key, &self.handle))?;

        // 计算grid和block
        let grid = grid_len.iter().product::<ArrayType>() as u32;
        let block = block_len.iter().product::<ArrayType>() as u32;

        // cuda 参数准备
        let block_len_total = block_len.iter().map(|x| *x as u32).product::<u32>();
        let src_block_stride = ArrayStruct::<ARRAY_SIZE>::new(src_block_stride.into_iter(), 0)?;
        let dst_block_stride = ArrayStruct::<ARRAY_SIZE>::new(dst_block_stride.into_iter(), 0)?;
        let src_grid_stride = ArrayStruct::<ARRAY_SIZE>::new(src_grid_stride.into_iter(), 0)?;
        let dst_grid_stride = ArrayStruct::<ARRAY_SIZE>::new(dst_grid_stride.into_iter(), 0)?;
        let block_len = ArrayStruct::<ARRAY_SIZE>::new(block_len.into_iter(), 1)?;
        let grid_len = ArrayStruct::<ARRAY_SIZE>::new(grid_len.into_iter(), 1)?;

        let filter_split_dims = split_dims
            .iter()
            .filter(|split_dim| split_dim.dim_len % split_dim.num_per_block != 0)
            .collect::<Vec<_>>();

        let constrains = match filter_split_dims.len() {
            0 => ArrayStruct([0; 8]),
            1 => ArrayStruct([
                filter_split_dims[0].array_struct_idx_grid,
                filter_split_dims[0].array_struct_idx_block,
                filter_split_dims[0].num_per_block as ArrayType,
                filter_split_dims[0].dim_len as ArrayType,
                0,
                0,
                0,
                0,
            ]),
            2 => ArrayStruct([
                filter_split_dims[0].array_struct_idx_grid,
                filter_split_dims[0].array_struct_idx_block,
                filter_split_dims[0].num_per_block as ArrayType,
                filter_split_dims[0].dim_len as ArrayType,
                filter_split_dims[1].array_struct_idx_grid,
                filter_split_dims[1].array_struct_idx_block,
                filter_split_dims[1].num_per_block as ArrayType,
                filter_split_dims[1].dim_len as ArrayType,
            ]),
            _ => unreachable!(),
        };

        let params = cuda::params![
            args.dst_base,
            args.src_base,
            block_dim,
            block_len_total,
            block_len,        // 各维度的长度
            src_block_stride, // 源tensor在各维度上的步长(bytes)
            dst_block_stride, // 目标tensor在各维度上的步长(bytes)
            grid_len,         // 各维度的长度
            src_grid_stride,  // 源tensor在各维度上的步长(bytes)
            dst_grid_stride,  // 目标tensor在各维度上的步长(bytes)
            constrains
        ];

        scheme.module.launch(
            &scheme.name,
            grid,
            block,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn kernel_name(unit_size: usize, constrain_num: usize) -> String {
    format!("rearrange_unit_{unit_size}_constrain_{constrain_num}")
}

fn kernel_code(unit_size: usize, constrain_num: usize, array_size: usize) -> String {
    let name = kernel_name(unit_size, constrain_num);
    // 根据 unit_size 选择对应的类型
    let tmem_type = match unit_size {
        1 => "uchar1",
        2 => "uchar2",
        4 => "float1",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => unreachable!(),
    };

    // 生成约束参数
    let constrain_param = match constrain_num {
        0 => String::new(),
        1 | 2 => format!(",const ArrayStruct<{constrain_num}, Constrains<ARRAY_TYPE>> constrains"),
        _ => unreachable!(),
    };

    // 生成共享内存声明
    let shared_mem_decl = match constrain_num {
        0 => String::new(),
        1 | 2 => format!("__shared__ int shared_constrains_grid_idx_multiple[{constrain_num}];"),
        _ => unreachable!(),
    };

    // 生成约束数组初始化代码
    let constrain_init = match constrain_num {
        0 => String::new(),
        1 | 2 => format!("int constrains_grid_idx_multiple[{constrain_num}] = {{0}};"),
        _ => unreachable!(),
    };

    // 生成约束检查代码
    let constrain_check = match constrain_num {
        0 => String::new(),
        _ => format!(
            r#"
            for (int j = 0; j < {constrain_num}; j++) {{
                if (i == constrains.a[j].grid_idx) {{
                    constrains_grid_idx_multiple[j] = idx * constrains.a[j].grid_div_block;
                }}
            }}"#
        ),
    };

    // 生成共享内存同步代码
    let shared_mem_sync = match constrain_num {
        0 => String::new(),
        _ => format!(
            r#"
            for (int j = 0; j < {constrain_num}; j++) {{
                shared_constrains_grid_idx_multiple[j] = constrains_grid_idx_multiple[j];
            }}"#
        ),
    };

    // 生成约束加载代码
    let constrain_load = match constrain_num {
        0 => String::new(),
        _ => format!(
            r#"
    int constrains_grid_idx_multiple[{constrain_num}];
    for (int j = 0; j < {constrain_num}; j++) {{
        constrains_grid_idx_multiple[j] = shared_constrains_grid_idx_multiple[j];
    }}"#
        ),
    };

    // 生成约束边界检查代码
    let constrain_boundary_check = match constrain_num {
        0 => String::new(),
        _ => format!(
            r#"
            for (int j = 0; j < {constrain_num}; j++) {{
                if (constrains.a[j].total_len != 0 && i == constrains.a[j].block_idx) {{
                    if (constrains_grid_idx_multiple[j] + idx >= constrains.a[j].total_len) {{
                        return;
                    }}
                }}
            }}"#
        ),
    };

    // 生成第一维度约束检查代码
    let first_dim_check = match constrain_num {
        0 => String::new(),
        _ => format!(
            r#"
    for (int j = 0; j < {constrain_num}; j++) {{
        if (constrains.a[j].total_len != 0 && 0 == constrains.a[j].block_idx) {{
            if (constrains_grid_idx_multiple[j] + remaining >= constrains.a[j].total_len) {{
                return;
            }}
        }}
    }}"#
        ),
    };

    format!(
        r#"
extern "C" __global__ void {name}(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,                    // block_len 各元素的乘积
    const ArrayStruct<{array_size}, ARRAY_TYPE> block_len,       // 各维度的长度
    const ArrayStruct<{array_size}, ARRAY_TYPE> src_block_stride,// 源tensor在各维度上的步长(bytes)
    const ArrayStruct<{array_size}, ARRAY_TYPE> dst_block_stride,// 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<{array_size}, ARRAY_TYPE> grid_len,        // 各维度的长度
    const ArrayStruct<{array_size}, ARRAY_TYPE> src_grid_stride, // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<{array_size}, ARRAY_TYPE> dst_grid_stride{constrain_param} // 目标tensor在各维度上的步长(bytes),切分维度的约束条件数组
             // 
) {{
    int remaining = threadIdx.x;
    if (remaining >= block_len_total) {{
        return;
    }}

    // 声明共享内存
    __shared__ int shared_src_offset;
    __shared__ int shared_dst_offset;
    {shared_mem_decl}

    if (threadIdx.x == 0) {{// 只让0号线程计算
        // 计算当前block处理的数据在src和dst中的基础偏移(bytes)
        int src_offset = 0;
        int dst_offset = 0;
        {constrain_init}
        int remaining = blockIdx.x;

        for (int i = {array_size} - 1; i >= 0; i--) {{
            int idx = remaining % grid_len.a[i];
            remaining /= grid_len.a[i];
            src_offset += idx * src_grid_stride.a[i];
            dst_offset += idx * dst_grid_stride.a[i];
            {constrain_check}
        }}
        
        // 将结果存入共享内存
        shared_src_offset = src_offset;
        shared_dst_offset = dst_offset;
        {shared_mem_sync}
    }}

    // 确保所有线程都能看到共享内存中的值
    __syncthreads();

    // 所有线程直接使用计算好的偏移值
    int src_offset = shared_src_offset;
    int dst_offset = shared_dst_offset;
    {constrain_load}

    for (int i = {array_size} - 1; i > 0; i--) {{
        if (block_len.a[i] > 1) {{
            int idx = remaining % block_len.a[i];
            remaining /= block_len.a[i];
            // 计算偏移量
            src_offset += idx * src_block_stride.a[i];
            dst_offset += idx * dst_block_stride.a[i];
            {constrain_boundary_check}
        }}
    }}

    src_offset += remaining * src_block_stride.a[0];
    dst_offset += remaining * dst_block_stride.a[0];
    {first_dim_check}

    // 执行数据拷贝，注意offset已经是字节偏移
    *reinterpret_cast<{tmem_type} *>(reinterpret_cast<char *>(dst) + dst_offset) =
        *reinterpret_cast<const {tmem_type} *>(reinterpret_cast<const char *>(src) + src_offset);
}}
"#
    )
}

fn format_code(unit_size: usize, constrain_num: usize) -> String {
    let mut code = String::new();

    // 添加头部定义
    code.push_str(&format!("#define ARRAY_SIZE {ARRAY_SIZE}\n"));
    code.push_str("#define ARRAY_TYPE int\n");
    code.push_str(CODE);
    code.push_str("\n");

    code.push_str(&kernel_code(unit_size, constrain_num, ARRAY_SIZE));
    code.push_str("\n\n");

    code
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
        use super::Scheme;
        use super::SchemeKey;

        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let op = Operator::new(&gpu);

        // 遍历所有可能的unit_size和constrain_num组合，编译所有kernel
        for unit_size in (0..=5).map(|n| (1 << n)) {
            for constrain_num in 0..=2 {
                println!(
                    "compile unit_size: {}, constrain_num: {}",
                    unit_size, constrain_num
                );
                let key = SchemeKey {
                    unit_size,
                    constrain_num,
                };
                op.schemes
                    .lock()
                    .unwrap()
                    .try_get_or_insert(key, || Scheme::new(key, &op.handle))
                    .unwrap();
            }
        }

        // 打印所有编译好的kernel信息
        gpu.apply(|ctx| {
            let schemes = op.schemes.lock().unwrap();
            for (key, scheme) in schemes.iter() {
                println!("{:?}", scheme.name);
                println!(
                    "unit_size: {}, constrain_num: {}\n{}",
                    key.unit_size,
                    key.constrain_num,
                    // scheme.name.to_str().unwrap(),
                    scheme.module.load(&scheme.name, ctx).info()
                );
                println!("----------------------------------------");
            }
        });
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

        let dt = ty::U64;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        gpu_op.scheme(&dyn_args(dt), 0).unwrap();

        const N: usize = 3;
        const TRANS_N: usize = 3;
        let shape: [usize; N] = [32, 2, 17];
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

        println!("s_src shape: {:?}", s_src.shape());
        println!("s_dst shape: {:?}", s_dst.shape());
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
}
