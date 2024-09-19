use super::{args::Meta, Args, Swiglu};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::{sizeof, gcd, get_or_err},
};
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    max_threads_block: usize,
    scheme: Option<Arc<ModuleBox>>,
}

const NAME: &str = "swiglu_f16";
const CODE: &str = include_str!("swiglu.cuh");

impl Swiglu<Gpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
            max_threads_block: handle.0.device().block_limit().max_threads,
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let Meta { dt, n: _, d: _ } = args.meta()?;
        if dt != F16 {
            todo!()
        }
        let cc = self.handle.device().compute_capability();
        self.scheme = Some(self.handle.compile_kernel(NAME, cc, || {
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
        }));
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta { dt, n, d } = args.meta()?;
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[sgn, sgd] = gate_layout.strides() else {
            unreachable!()
        };
        let &[sun, sud] = up_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(sgn);
        get_or_err!(sgd);
        get_or_err!(sun);
        get_or_err!(sud);

        let unit = sizeof!(dt)? as isize;
        if sgd != unit || sud != unit {
            return Err(locate_error!("Unsupported layout"));
        };

        let Some(m) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };

        let sg = (sgn / unit) as i32;
        let su = (sun / unit) as i32;
        let params = cuda::params![gate_base, sg, up_base, su];
        let block = gcd(self.max_threads_block, d);

        m.launch(
            CString::new(NAME).unwrap(),
            (n as _, (d / block) as _),
            block as u32,
            params.as_ptr(),
            0,
            queue,
        );
        Ok(())
    }
}

#[test]
fn test() {}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use common::{dyn_, Handle, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Handle>(dt: DigitLayout) -> Args<H> {
        use std::ptr::{null, null_mut};
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        }
    }

    fn args<H: Handle>(
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
        op.scheme(&dyn_args(F16)).unwrap();

        let module = op.scheme.as_ref().unwrap();
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
        use crate::{
            common_cpu::{Handle as Cpu, ThisThread},
            nvidia_gpu::cast_load,
            utils::{Diff, ErrorCollector},
        };
        use cuda::memcpy_d2h;
        use half::f16;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(F64)).unwrap();
        gpu_op.scheme(&dyn_args(F16)).unwrap();

        let n = 5632;
        let d = 2048;

        let mut gate = vec![0.0f64; n * d];
        let mut up = vec![0.0f64; n * d];
        rand::thread_rng().fill(&mut gate[..]);
        rand::thread_rng().fill(&mut up[..]);
        let up = up;

        let gate_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut gate = cast_load(&gate, |&x| f16::from_f64(x), &stream);
            let up = cast_load(&up, |&x| f16::from_f64(x), &stream);
            gpu_op
                .launch(
                    &args(F16, n, d, gate.as_mut_ptr().cast(), up.as_ptr().cast()),
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
                &ThisThread,
            )
            .unwrap();

        let diff = gate_ref
            .into_par_iter()
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
