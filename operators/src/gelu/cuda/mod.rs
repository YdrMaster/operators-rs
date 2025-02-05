use super::{args::Meta, Args, Gelu};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    get_static, strides_not_support, type_not_support,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::types::F16;
use std::{
    ffi::{c_uint, CString},
    sync::Arc,
};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
    module: Arc<ModuleBox>,
}

const NAME: &str = "gelu_f16";
const CODE: &str = include_str!("gelu.cuh");
impl Gelu<Gpu> for Operator {}

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
        let Args { layout, base } = args;
        if dt != F16 {
            return Err(type_not_support("").into());
        }
        let &[_, ds] = layout.strides() else {
            unreachable!()
        };

        get_static! {
             n   d  ds
        }

        let unit = dt.nbytes() as isize;
        if ds != unit {
            return Err(strides_not_support("").into());
        };

        let params = cuda::params![base];
        let block = gcd(self.max_threads_block, d);

        self.module.launch(
            CString::new(NAME).unwrap(),
            (n * d).div_ceil(block) as c_uint,
            block as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn format_code() -> String {
    format!(
        r#"{CODE}

extern "C" __global__ void {NAME}(
    half *__restrict__ data
){{
    gelu(data);
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
        use std::ptr::null_mut;
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            layout: layout.clone(),
            base: null_mut(),
        }
    }
    fn args<H: Hardware>(dt: DigitLayout, n: usize, d: usize, base: *mut H::Byte) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt, &[n, d]);
        Args {
            layout: layout.clone(),
            base,
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

        let n = 1024;
        let d = 2048;

        let mut data = vec![0.0f64; n * d];
        rand::rng().fill(&mut data[..]);

        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut data = cast_load(&data, f16::from_f64, &stream);
            gpu_op
                .launch(&args(F16, n, d, data.as_mut_ptr().cast()), &mut [], &stream)
                .unwrap();
            let mut host = vec![f16::ZERO; n * d];
            memcpy_d2h(&mut host, &data);
            host
        });

        let mut data_ref = data;
        cpu_op
            .launch(
                &args(F64, n, d, data_ref.as_mut_ptr().cast()),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = data_ref
            .into_iter()
            .zip(data_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
