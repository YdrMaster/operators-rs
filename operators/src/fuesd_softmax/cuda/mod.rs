use super::{
    args::{AttnMask, Meta},
    Args, FusedSoftmax,
};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    get_static, strides_not_support, type_not_support, ByteOf, LaunchError, QueueAlloc,
    SchemeError,
};
use digit_layout::types::F16;
use std::{
    collections::HashMap,
    ffi::{c_float, CString},
    mem::size_of,
    sync::Arc,
};

pub struct Operator {
    _handle: Arc<Handle>,
    scheme: HashMap<AttnMask, Scheme>,
}

impl FusedSoftmax<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            _handle: node.0.clone(),
            scheme: [AttnMask::Causal]
                .map(|mask| (mask, Scheme::new(&node.0, mask)))
                .into_iter()
                .collect(),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt } = args.meta()?;
        if dt == F16 {
            Ok(0)
        } else {
            Err(type_not_support(""))
        }
    }

    fn launch<T>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue: &T,
    ) -> Result<(), LaunchError>
    where
        T: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt } = args.meta()?;
        let Args {
            att_mask,
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
            return Err(type_not_support("").into());
        }

        get_static! {
            nh seq_len att_len
            sh ss      sa
        }

        let unit = dt.nbytes() as isize;
        if sa != unit {
            return Err(strides_not_support("").into());
        };

        let scheme = &self.scheme[att_mask];
        let grid_dims = (nh as u32, seq_len as u32);
        let block_size = scheme.max_threads_block as u32;
        let sh = (sh / unit) as i32;
        let ss = (ss / unit) as i32;
        let att_len = att_len as u32;
        let params = cuda::params![att_base, 0i32, sh, ss, att_len];

        if att_len <= block_size {
            scheme.module.launch(
                &scheme.padding,
                grid_dims,
                att_len,
                params.as_ptr(),
                0,
                queue.queue(),
            );
        } else {
            let num_items_thread = att_len.div_ceil(block_size);
            let smem = (num_items_thread * block_size) as usize;
            scheme.module.launch(
                &scheme.folding,
                grid_dims,
                block_size,
                params.as_ptr(),
                smem * size_of::<c_float>(),
                queue.queue(),
            );
        }

        Ok(())
    }
}

struct Scheme {
    max_threads_block: usize,
    padding: CString,
    folding: CString,
    module: Arc<ModuleBox>,
}

impl Scheme {
    fn new(handle: &Arc<Handle>, mask: AttnMask) -> Self {
        const NAME: &str = "fused_softmax";
        const CODE: &str = include_str!("fused_softmax.cuh");

        let mask = match mask {
            AttnMask::None => "AttentionNonMask",
            AttnMask::Causal => "AttentionCausalMask",
        };
        let device = handle.device();
        let max_threads_block = device.block_limit().max_threads;
        let cc = device.compute_capability();
        let padding = format!("fused_softmax_padding_{max_threads_block}");
        let folding = format!("fused_softmax_folding_{max_threads_block}");

        let module = handle.compile_kernel(NAME, cc, || {
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
    folding<{max_threads_block}>
    (att, {mask}(), att_len, stride_z, stride_y, stride_x);
}}
"#
            )
        });
        Self {
            max_threads_block,
            padding: CString::new(padding).unwrap(),
            folding: CString::new(folding).unwrap(),
            module,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, AttnMask, Gpu, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::null_mut;
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]),
            att_base: null_mut(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        seq_len: usize,
        att_len: usize,
        att_base: *mut H::Byte,
    ) -> Args<H> {
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_contiguous(dt, &[nh, seq_len, att_len]),
            att_base,
        }
    }

    #[test]
    fn test_compile() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        op.scheme(&dyn_args(ty::F16), 0).unwrap();

        gpu.apply(|ctx| {
            for (mask, scheme) in op.scheme {
                println!("{mask:?}============================");
                println!("{}", scheme.padding.to_str().unwrap());
                println!("{}", scheme.module.load(&scheme.padding, ctx).info());
                println!("{}", scheme.folding.to_str().unwrap());
                println!("{}", scheme.module.load(&scheme.folding, ctx).info());
            }
        })
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            cuda::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use cuda::memcpy_d2h;
        use half::f16;
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(ty::F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(ty::F16), 0).unwrap();

        let nh = 32;
        for (seq_len, att_len) in [(1, 511), (1, 2048), (7, 511), (7, 2048)] {
            let mut att = vec![0.0f64; nh * seq_len * att_len];
            rand::rng().fill(&mut att[..]);

            let att_ans = gpu.apply(|ctx| {
                let stream = ctx.stream();
                let mut att = cast_load(&att, f16::from_f64, &stream);
                gpu_op
                    .launch(
                        &args(ty::F16, nh, seq_len, att_len, att.as_mut_ptr().cast()),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; nh * seq_len * att_len];
                memcpy_d2h(&mut host, &att);
                host
            });

            let mut att_ref = att;
            cpu_op
                .launch(
                    &args(ty::F64, nh, seq_len, att_len, att_ref.as_mut_ptr().cast()),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = att_ref
                .into_iter()
                .zip(att_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
