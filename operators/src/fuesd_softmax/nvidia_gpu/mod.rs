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
    pub max_seq_len: usize,
    pub max_num_threads_block: usize,
    pub compute_capability: ComputeCapability,
}

impl Config {
    pub fn config_for(dev: &cuda::Device, data_layout: DataLayout, max_seq_len: usize) -> Self {
        Self {
            data_layout,
            max_seq_len,
            max_num_threads_block: dev.max_block_dims().0,
            compute_capability: dev.compute_capability(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operator {
    global: __global__,
    padding: CString,
    folding: CString,
    max_num_threads_block: usize,
}

impl common::Operator<Gpu> for Operator {
    type Config = Config;
    type Error = ErrorPosition;
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        let &Self::Config {
            data_layout,
            max_seq_len,
            max_num_threads_block,
            compute_capability,
        } = config;

        const CODE: &str = include_str!("fused_softmax.cuh");

        let mask = "AttentionCausualMask";
        let max_num_items_thread =
            (max_seq_len + max_num_threads_block - 1) / max_num_threads_block;
        let padding = format!("fused_softmax_padding_{max_num_threads_block}");
        let folding =
            format!("fused_softmax_folding_{max_num_threads_block}x{max_num_items_thread}");

        if data_layout != F16 {
            return Err(locate_error!());
        }
        let code = format!(
            r#"{CODE}

extern "C" __global__ void {padding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x
){{
    padding<{max_num_threads_block}>
    (att, {mask}(), stride_z, stride_y, stride_x);
}}

extern "C" __global__ void {folding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x,

    unsigned int const att_len
){{
    folding<{max_num_threads_block}, {max_num_items_thread}>
    (att, {mask}(), att_len, stride_z, stride_y, stride_x);
}}
"#
        );

        let (ptx, log) = Ptx::compile(code, compute_capability);
        let ptx = ptx
            .map_err(|e| locate_error!(format!("Failed to compile fused_softmax: {e:?}\n{log}")))?;
        if !log.is_empty() {
            warn!("{log}");
        }
        Ok(Self {
            global: __global__::load(&ptx),
            padding: CString::new(padding).unwrap(),
            folding: CString::new(folding).unwrap(),
            max_num_threads_block,
        })
    }
}

pub struct Scheme {
    global: __global__,
    name: CString,

    grid: (c_uint, c_uint),
    block: c_uint,
    stride_head: c_int,
    stride_token: c_int,
    offset: usize,

    att_len: c_uint,
}

impl common::Scheme<Gpu, Operator> for Scheme {
    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        let SchemeLayout {
            nh,
            seq_len,
            att_len,
            stride_head,
            stride_token,
            offset,
        } = SchemeLayout::new(F16, layout)?;

        let (name, block) = if att_len <= op.max_num_threads_block {
            (op.padding.clone(), att_len)
        } else {
            // FIXME: 极度怪异的行为。
            // 如果 block dims 不取 self.block_size, kernel 会在随机位置计算出错误数据。
            // 然而，如果打印 block dims，计算就不会出错。只能打印，写入带内存屏障的原子变量、锁、Flush 均无效。
            // 现在这样浪费了一些线程。
            // let mut block_dims = 0;
            // for items_per_thread in 2.. {
            //     block_dims = (att_len + items_per_thread - 1) / items_per_thread;
            //     block_dims = (block_dims + 31) / 32 * 32;
            //     if block_dims <= self.block_size {
            //         break;
            //     }
            // }
            (op.folding.clone(), op.max_num_threads_block)
        };
        // println!("block dims = {block_dims}");

        Ok(Self {
            global: op.global,
            name,

            grid: (nh as _, seq_len as _),
            block: block as _,
            stride_head: stride_head as _,
            stride_token: stride_token as _,
            offset,

            att_len: att_len as _,
        })
    }

    type Params<'ctx> = Params<Gpu>;
    fn launch(&self, att: &Self::Params<'_>, queue: &QueueOf<Gpu>) {
        let att = unsafe { att.add(self.offset) };
        let params = cuda::params![att, 0i32, self.stride_head, self.stride_token, self.att_len];
        self.global
            .launch(&self.name, self.grid, self.block, params.as_ptr(), 0, queue);
    }
}
