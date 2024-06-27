use super::args::{Args, Meta};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{locate_error, ErrorPosition, QueueOf};
use cuda::Version;
use digit_layout::types::F16;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<(Scheme, Arc<ModuleBox>)>,
}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let Meta { dt } = args.meta()?;
        if dt != F16 {
            todo!()
        }
        self.scheme(
            self.handle.device().compute_capability(),
            *args.att_layout.shape()[2].get_static().unwrap(),
        )
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta { dt } = args.meta()?;
        let Args {
            att_layout,
            att_base,
        } = args;
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        let &[sh, ss, sa] = att_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(nh);
        get_or_err!(seq_len);
        get_or_err!(att_len);
        get_or_err!(sh);
        get_or_err!(ss);
        get_or_err!(sa);

        todo!()
    }
}

struct Scheme {
    max_items_thread: usize,
    padding: CString,
    folding: CString,
}

const CODE: &str = include_str!("fused_softmax.cuh");
impl Operator {
    fn scheme(&mut self, cc: Version, seq_len: usize) -> Result<(), ErrorPosition> {
        let name = format!("fuse_softmax_{seq_len}");

        let mask = "AttentionCausualMask";
        let max_threads_block = self.handle.device().block_limit().max_threads;
        let padding = format!("fused_softmax_padding_{max_threads_block}");
        let max_items_thread = (seq_len + max_threads_block - 1) / max_threads_block;
        let folding = format!("fused_softmax_folding_{max_threads_block}x{max_items_thread}");

        let module = self
            .handle
            .compile(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {padding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x
){{
    padding<{max_threads_block}>
    (att, {mask}(), stride_z, stride_y, stride_x);
}}

extern "C" __global__ void {folding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x,

    unsigned int const att_len
){{
    folding<{max_threads_block}, {max_items_thread}>
    (att, {mask}(), att_len, stride_z, stride_y, stride_x);
}}
"#
                )
            })
            .map_err(|(e, log)| locate_error!(format!("Failed to compile {name}: {e:?}\n{log}")))?;
        self.scheme = Some((
            Scheme {
                max_items_thread,
                padding: CString::new(padding).unwrap(),
                folding: CString::new(folding).unwrap(),
            },
            module,
        ));
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use std::ptr::null_mut;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    for num_items_thread in 1..=16 {
        <Operator as common::Operator>::scheme(
            &mut op,
            &Args {
                att_layout: TensorLayout::new(
                    F16,
                    &[dyn_(), dyn_(), (num_items_thread * 1024).into()],
                    &[dyn_(); 3],
                ),
                att_base: null_mut(),
            },
        )
        .unwrap();
        let (scheme, module) = op.scheme.as_ref().unwrap();
        handle.apply(|ctx| {
            println!("============================");
            // println!("{}", scheme.padding.to_str().unwrap());
            // println!("{}", module.load(&scheme.padding, ctx).info());
            println!("{}", scheme.folding.to_str().unwrap());
            println!("{}", module.load(&scheme.folding, ctx).info());
        })
    }
}

// pub struct Scheme {
//     global: __global__,
//     name: CString,

//     grid: (c_uint, c_uint),
//     block: c_uint,
//     stride_head: c_int,
//     stride_token: c_int,
//     offset: usize,

//     att_len: c_uint,
// }

// impl FuesdSoftmax<Gpu> for Scheme {}

// impl common::Scheme for Scheme {
//     type Device = Gpu;
//     type Operator = Operator;

//     type LayoutAttrs = LayoutAttrs;
//     type Error = ErrorPosition;
//     fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
//         let SchemeLayout {
//             nh,
//             seq_len,
//             att_len,
//             stride_head,
//             stride_token,
//             offset,
//         } = SchemeLayout::new(F16, layout)?;

//         let (name, block) = if att_len <= op.max_num_threads_block {
//             (op.padding.clone(), att_len)
//         } else {
//             // FIXME: 极度怪异的行为。
//             // 如果 block dims 不取 self.block_size, kernel 会在随机位置计算出错误数据。
//             // 然而，如果打印 block dims，计算就不会出错。只能打印，写入带内存屏障的原子变量、锁、Flush 均无效。
//             // 现在这样浪费了一些线程。
//             // let mut block_dims = 0;
//             // for items_per_thread in 2.. {
//             //     block_dims = (att_len + items_per_thread - 1) / items_per_thread;
//             //     block_dims = (block_dims + 31) / 32 * 32;
//             //     if block_dims <= self.block_size {
//             //         break;
//             //     }
//             // }
//             (op.folding.clone(), op.max_num_threads_block)
//         };
//         // println!("block dims = {block_dims}");

//         Ok(Self {
//             global: op.global,
//             name,

//             grid: (nh as _, seq_len as _),
//             block: block as _,
//             stride_head: stride_head as _,
//             stride_token: stride_token as _,
//             offset,

//             att_len: att_len as _,
//         })
//     }

//     type Params = Params<Gpu>;
//     fn launch(&self, att: &Self::Params, queue: &QueueOf<Gpu>) {
//         let att = unsafe { att.add(self.offset) };
//         let params = cuda::params![att, 0i32, self.stride_head, self.stride_token, self.att_len];
//         self.global
//             .launch(&self.name, self.grid, self.block, params.as_ptr(), 0, queue);
//     }
// }
