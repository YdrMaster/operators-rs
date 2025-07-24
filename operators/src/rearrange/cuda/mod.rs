use super::{Args, Rearrange, args::Scheme as ArgsScheme};
use crate::rank_not_support;
use crate::{
    ByteOf, LaunchError, QueueAlloc, SchemeDiversity,
    cuda::{Gpu, Handle, ModuleBox},
};
use itertools::Itertools;
use lru::LruCache;
use std::cmp::max;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    unit_size: usize,
    block_array_size: usize,
    grid_array_size: usize,
    constrain_num: usize,
}

#[derive(Clone)]
struct Scheme {
    module: Arc<ModuleBox>,
    name: CString,
}

impl Scheme {
    pub fn new(key: SchemeKey, handle: &Arc<Handle>) -> Self {
        let name = kernel_name(key);
        let cc = handle.device().compute_capability();
        // for DEBUG
        // let code = format_code(key.unit_size, key.constrain_num);
        // std::fs::write("rearrange.cu", code).unwrap();

        Self {
            module: handle.compile_kernel(&name, cc, || format_code(key)),
            name: CString::new(name).unwrap(),
        }
    }
}

/// Type used for array indices and strides
type ArrayType = i32;

// 默认的数组大小，同时也是最大的数组大小，不能为0
const DEFAULT_ARRAY_SIZE: usize = 5;
const CONSTRAIN_ARRAY_SIZE: usize = 8;

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
struct ArrayStruct(Vec<ArrayType>);

impl ArrayStruct {
    fn new(mut array: Vec<ArrayType>, default: ArrayType) -> Self {
        while array.len() < DEFAULT_ARRAY_SIZE {
            array.push(default);
        }
        Self(array)
    }

    fn try_into_array<const N: usize>(self) -> Result<[ArrayType; N], LaunchError> {
        let ArrayStruct(vec) = self;
        if vec.len() <= N {
            Ok(std::array::from_fn(|i| vec.get(i).copied().unwrap_or(0)))
        } else {
            Err(rank_not_support("over length"))
        }
    }
}

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
        if scheme_update.ndim() == 0 {
            let unit = scheme_update.unit();
            let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let src = unsafe { from_raw_parts(args.src_base, unit) };
            queue_alloc.queue().memcpy_d2d(dst, src);
            return Ok(());
        }

        let src_strides = scheme_update.src_strides();
        let dst_strides = scheme_update.dst_strides();
        let shape = scheme_update.shape().collect::<Vec<_>>();
        let unit = scheme_update.unit();
        let ndim = scheme_update.ndim();

        //src strides 降序 index
        let src_strides_desc_idx = (0..scheme_update.ndim())
            .zip(src_strides)
            .sorted_by(|a, b| b.1.cmp(a.1))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        //分离维度，分成grid处理的维度和block处理的维度，与dst的维度相对应
        let mut block_dim_choose = vec![false; ndim];

        //TODO 需要优化
        let max_block_size = 256;
        let mut split_dims = Vec::new(); // 长度最多为2

        //进行维度选择
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

        let mut block_len = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut src_block_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut dst_block_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);

        let mut grid_len = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut src_grid_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut dst_grid_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);

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
        let mut grid_dim = 0_u32;
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

        let filter_split_dims = split_dims
            .iter()
            .filter(|split_dim| split_dim.dim_len % split_dim.num_per_block != 0)
            .collect::<Vec<_>>();

        let constrain_num = filter_split_dims.len();

        // 准备kernel
        let key = SchemeKey {
            unit_size: unit,
            constrain_num,
            block_array_size: block_len.len(),
            grid_array_size: grid_len.len(),
        };

        let mut schemes = self.schemes.lock().unwrap();

        let scheme = schemes.get_or_insert(key, || Scheme::new(key, &self.handle));

        // 计算grid和block
        let grid = grid_len.iter().product::<ArrayType>() as u32;
        let block = block_len.iter().product::<ArrayType>() as u32;

        // cuda 参数准备
        let block_len_total = block_len.iter().map(|x| *x as u32).product::<u32>();
        let src_block_stride = ArrayStruct::new(src_block_stride, 0);
        let dst_block_stride = ArrayStruct::new(dst_block_stride, 0);
        let src_grid_stride = ArrayStruct::new(src_grid_stride, 0);
        let dst_grid_stride = ArrayStruct::new(dst_grid_stride, 0);
        let block_len = ArrayStruct::new(block_len, 1);
        let grid_len = ArrayStruct::new(grid_len, 1);

        let constrains = match filter_split_dims.len() {
            0 => ArrayStruct(vec![0; 8]),
            1 => ArrayStruct(vec![
                filter_split_dims[0].array_struct_idx_grid,
                filter_split_dims[0].array_struct_idx_block,
                filter_split_dims[0].num_per_block as ArrayType,
                filter_split_dims[0].dim_len as ArrayType,
                0,
                0,
                0,
                0,
            ]),
            2 => ArrayStruct(vec![
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
            block_len.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 各维度的长度
            src_block_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 源tensor在各维度上的步长(bytes)
            dst_block_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 目标tensor在各维度上的步长(bytes)
            grid_len.try_into_array::<DEFAULT_ARRAY_SIZE>()?,         // 各维度的长度
            src_grid_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 源tensor在各维度上的步长(bytes)
            dst_grid_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 目标tensor在各维度上的步长(bytes)
            constrains.try_into_array::<CONSTRAIN_ARRAY_SIZE>()?
        ];

        scheme.module.launch(
            &scheme.name,
            (grid, block, 0),
            &params.to_ptrs(),
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn kernel_name(
    SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    }: SchemeKey,
) -> String {
    let tmem_type = match unit_size {
        1 => "uchar1",
        2 => "uchar2",
        4 => "float1",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => unreachable!(),
    };
    format!(
        "rearrange_unit_{tmem_type}_block_{block_array_size}_grid_{grid_array_size}_constrain_{constrain_num}"
    )
}

fn format_code(
    SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    }: SchemeKey,
) -> String {
    assert!(block_array_size != 0);

    let kernel_name = kernel_name(SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    });
    //处理 grid_array_size = 0的情况
    let grid_array_size = max(grid_array_size, 1);

    let mut code = String::new();

    let tmem_type = match unit_size {
        1 => "uchar1",
        2 => "uchar2",
        4 => "float1",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => unreachable!(),
    };

    // 添加头部定义
    code.push_str(&format!("#define BLOCK_ARRAY_SIZE {block_array_size}\n"));
    code.push_str(&format!("#define GRID_ARRAY_SIZE {grid_array_size}\n"));
    code.push_str("#define ARRAY_TYPE int\n");
    code.push_str(&format!("#define CONSTRAIN_NUM {constrain_num}\n"));
    code.push_str(CODE);
    code.push('\n');

    // 添加实例化宏调用
    code.push_str(&format!(
        r#"
extern "C" __global__ void {kernel_name}(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> block_len,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> src_block_stride,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> grid_len,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> src_grid_stride,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride
#if CONSTRAIN_NUM > 0
    ,const ArrayStruct<CONSTRAIN_NUM, Constrains<ARRAY_TYPE>> constrains
#endif
) {{
    rearrange_kernel<{tmem_type}, {constrain_num}>(
        dst, src, block_dim, block_len_total,
        block_len, src_block_stride, dst_block_stride,
        grid_len, src_grid_stride, dst_grid_stride
#if CONSTRAIN_NUM > 0
        ,constrains
#endif
    );
}}
"#
    ));
    code.push('\n');

    code
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use super::{Args, Gpu, Operator};
    use crate::{ConstPtr, Hardware, MutPtr, Operator as _, TensorLayout};
    use digit_layout::{DigitLayout, types as ty};
    use log::debug;

    // fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
    //     use std::ptr::{null, null_mut};
    //     Args {
    //         dst_layout: TensorLayout::new(dt, &[0; 2], &[0; 2]),
    //         dst_base: null_mut(),
    //         src_layout: TensorLayout::new_dyn(dt, &[0; 2], &[0; 2]),
    //         src_base: null(),
    //     }
    // }

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
                    block_array_size: 5,
                    grid_array_size: 5,
                };
                op.schemes
                    .lock()
                    .unwrap()
                    .get_or_insert(key, || Scheme::new(key, &op.handle));
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

    fn copute_with_check<const N: usize, const TRANS_N: usize>(
        gpu: &Gpu,
        shape: [usize; N],
    ) -> Duration {
        assert!(TRANS_N <= N, "TRANS_N must be less than or equal to N");
        use super::super::common_cpu::Operator as RefOp;
        use crate::common_cpu::{Cpu, ThisThread};

        use cuda::memcpy_d2h;
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;

        let dt = ty::U64;

        let cpu_op = RefOp::new(&Cpu);
        let gpu_op = Operator::new(gpu);

        let mut r_shape = shape;
        r_shape[0..TRANS_N].reverse();

        let trans_param: [usize; TRANS_N] =
            (0..TRANS_N).rev().collect::<Vec<_>>().try_into().unwrap();

        let mut src = vec![0u64; shape.iter().product::<usize>()];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<N>::new_contiguous(&shape, BigEndian, ele);
        let s_dst =
            ArrayLayout::<N>::new_contiguous(&r_shape, BigEndian, ele).transpose(&trans_param);

        debug!("s_src shape: {:?}", s_src.shape());
        debug!("s_dst shape: {:?}", s_dst.shape());
        debug!("s_src strides: {:?}", s_src.strides());
        debug!("s_dst strides: {:?}", s_dst.strides());

        let (dst_ans, time) = gpu.apply(|ctx| {
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

            let mut host = vec![0u64; shape.iter().product::<usize>()];
            memcpy_d2h(&mut host, &dst);
            (host, time)
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
        time
    }

    #[test]
    fn test_compute() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        let shape = [2];
        let time = copute_with_check::<1, 1>(&gpu, shape);
        println!("time: {time:?}");

        let shape = [13];
        let time = copute_with_check::<1, 1>(&gpu, shape);
        println!("time: {time:?}");

        let shape = [16, 2, 16];
        let time = copute_with_check::<3, 3>(&gpu, shape);
        println!("time: {time:?}");

        let shape = [32, 2, 17];
        let time = copute_with_check::<3, 3>(&gpu, shape);
        println!("time: {time:?}");

        let shape = [32, 2, 17, 2, 13];
        let time = copute_with_check::<5, 5>(&gpu, shape);
        println!("time: {time:?}");
    }
}
