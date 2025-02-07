use super::{args::Scheme, Add, Args};
use crate::{
    cuda::{dt_name, Gpu, Handle, ModuleBox},
    shape_not_support, strides_not_support,
    utils::{gcd, type_distinct},
    ByteOf, LaunchError, QueueAlloc, SchemeDiversity, SchemeError,
};
use digit_layout::DigitLayout;
use lru::LruCache;
use std::{
    ffi::{c_uint, CString},
    sync::{Arc, Mutex},
};

pub struct Operator {
    handle: Arc<Handle>,
    max_threads_block: usize,
    schemes: Mutex<LruCache<DigitLayout, Arc<ModuleBox>>>,
}
impl Add<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
            max_threads_block: node.0.device().block_limit().max_threads,
            schemes: node.0.scheme_cache(SchemeDiversity::Low),
        }
    }

    #[inline]
    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let dt = type_distinct(&[args.c_layout.dt(), args.a_layout.dt(), args.b_layout.dt()])?;
        self.schemes
            .lock()
            .unwrap()
            .get_or_insert(dt, || compile(&self.handle, dt));
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
        let dt = scheme.dt();
        let count = scheme.count();

        let &[1] = scheme.idx_strides() else {
            return Err(shape_not_support("").into());
        };
        let &[sc] = scheme.c_strides() else {
            return Err(shape_not_support("").into());
        };
        let &[sa] = scheme.a_strides() else {
            return Err(shape_not_support("").into());
        };
        let &[sb] = scheme.b_strides() else {
            return Err(shape_not_support("").into());
        };
        let unit = dt.nbytes() as isize;
        if sc != unit || sa != unit || sb != unit {
            return Err(strides_not_support("").into());
        }

        let block_dims = gcd(count, self.max_threads_block);
        let grid_dims = count / block_dims;
        let Args {
            c_base,
            a_base,
            b_base,
            ..
        } = args;
        let params = cuda::params![c_base, a_base, b_base];

        self.schemes
            .lock()
            .unwrap()
            .get_or_insert(dt, || compile(&self.handle, dt))
            .launch(
                CString::new("add").unwrap(),
                grid_dims as c_uint,
                block_dims as c_uint,
                params.as_ptr(),
                0,
                queue_alloc.queue(),
            );
        Ok(())
    }
}

fn compile(handle: &Arc<Handle>, dt: DigitLayout) -> Arc<ModuleBox> {
    const CODE: &str = include_str!("add.cuh");
    let cc = handle.device().compute_capability();
    let ty = dt_name(dt);
    handle.compile_kernel(format!("add_{ty}"), cc, || {
        format!(
            r#"{CODE}

extern "C" __global__ void add(
    {ty} *__restrict__ c,
    {ty} const *__restrict__ a,
    {ty} const *__restrict__ b
){{
    _add(c, a, b);
}}"#
        )
    })
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };
    use std::ptr::null;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::null_mut;
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            c_layout: layout.clone(),
            c_base: null_mut(),
            a_layout: layout.clone(),
            a_base: null(),
            b_layout: layout.clone(),
            b_base: null(),
        }
    }
    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,

        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[n, d]),
            c_base,
            a_layout: TensorLayout::new_contiguous(dt, &[n, d]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[n, d]),
            b_base,
        }
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

        let n = 1;
        let d = 768;
        let len = n * d;
        let mut c = vec![0.0f64; len];
        let mut a = vec![0.1f64; len];
        let mut b = vec![0.1f64; len];
        rand::rng().fill(&mut a[..]);
        rand::rng().fill(&mut b[..]);
        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let mut c = rt.malloc::<f16>(c.len());
            let a = cast_load(&a, f16::from_f64, &stream);
            let b = cast_load(&b, f16::from_f64, &stream);
            gpu_op
                .launch(
                    &args(
                        F16,
                        n,
                        d,
                        c.as_mut_ptr().cast(),
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; len];
            memcpy_d2h(&mut host, &c);
            host
        });
        cpu_op
            .launch(
                &args(
                    F64,
                    n,
                    d,
                    c.as_mut_ptr().cast(),
                    a.as_ptr().cast(),
                    b.as_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        let diff = c
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
