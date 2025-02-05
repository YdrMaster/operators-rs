use super::{args::Meta, Args, Swiglu};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    get_static, strides_not_support, type_not_support,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::types::F16;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
    module: Arc<ModuleBox>,
}

const NAME: &str = "swiglu_f16";

impl Swiglu<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        let device = node.0.device();
        Self {
            _handle: node.0.clone(),
            max_threads_block: device.block_limit().max_threads,
            module: node
                .0
                .compile_kernel(NAME, device.compute_capability(), format_code),
        }
    }

    #[inline]
    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
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
        let Meta { dt, n, d } = args.meta()?;
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[gns, gds] = gate_layout.strides() else {
            unreachable!()
        };
        let &[uns, uds] = up_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(type_not_support("").into());
        }

        get_static! {
             n   d
            gns gds
            uns uds
        }

        let unit = dt.nbytes() as isize;
        if gds != unit || uds != unit {
            return Err(strides_not_support("").into());
        };

        let sg = (gns / unit) as i32;
        let su = (uns / unit) as i32;
        let params = cuda::params![gate_base, sg, up_base, su];
        let block = gcd(self.max_threads_block, d);

        self.module.launch(
            CString::new(NAME).unwrap(),
            (n as _, (d / block) as _),
            block as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn format_code() -> String {
    const CODE: &str = include_str!("swiglu.cuh");
    format!(
        r#"{CODE}

extern "C" __global__ void {NAME}(
    half *__restrict__ gate,
    int const stride_gate,
    half const *__restrict__ up,
    int const stride_up
){{
    swiglu(gate, stride_gate, up, stride_up);
}}"#
    )
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::{null, null_mut};
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,
        gate_base: *mut H::Byte,
        up_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt, &[n, d]);
        Args {
            gate_layout: layout.clone(),
            gate_base,
            up_layout: layout,
            up_base,
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
        op.scheme(&dyn_args(F16), 0).unwrap();

        gpu.apply(|ctx| {
            println!(
                "{NAME}\n{}",
                op.module.load(CString::new(NAME).unwrap(), ctx).info()
            );
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
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16), 0).unwrap();

        let n = 5632;
        let d = 2048;

        let mut gate = vec![0.0f64; n * d];
        let mut up = vec![0.0f64; n * d];
        rand::rng().fill(&mut gate[..]);
        rand::rng().fill(&mut up[..]);
        let up = up;

        let gate_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut gate = cast_load(&gate, f16::from_f64, &stream);
            let up = cast_load(&up, f16::from_f64, &stream);
            gpu_op
                .launch(
                    &args(F16, n, d, gate.as_mut_ptr().cast(), up.as_ptr().cast()),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; n * d];
            memcpy_d2h(&mut host, &gate);
            host
        });

        let mut gate_ref = gate;
        cpu_op
            .launch(
                &args(F64, n, d, gate_ref.as_mut_ptr().cast(), up.as_ptr().cast()),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = gate_ref
            .into_iter()
            .zip(gate_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
