use super::{Add, Args};
use crate::{
    add::args::Meta,
    cuda::{dt_name, Gpu, Handle, ModuleBox},
    get_static, strides_not_support,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeDiversity, SchemeError,
};
use digit_layout::DigitLayout;
use lru::LruCache;
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

pub struct Operator {
    handle: Arc<Handle>,
    max_threads_block: usize,
    schemes: Mutex<LruCache<SchemeKey, Scheme>>,
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
        let Meta { dt, .. } = args.meta()?;

        let key = SchemeKey { dt };
        self.schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?;
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
            c_layout,
            c_base,
            a_layout,
            a_base,
            b_layout,
            b_base,
        } = args;
        let &[cns, cds] = c_layout.strides() else {
            unreachable!()
        };
        let &[_, ads] = a_layout.strides() else {
            unreachable!()
        };
        let &[_, bds] = b_layout.strides() else {
            unreachable!()
        };

        get_static!(
            n   d
            cns cds
            ads
            bds
        );

        let unit = dt.nbytes() as isize;
        if cds != unit || ads != unit || bds != unit {
            return Err(strides_not_support("").into());
        };

        let key = SchemeKey { dt };
        let scheme = self
            .schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?
            .clone();

        let block = gcd(self.max_threads_block, d);
        let cns = (cns / unit) as i32;
        let params = cuda::params![c_base, a_base, b_base, cns];
        scheme.module.launch(
            &scheme.name,
            (n as u32, (d / block) as u32),
            block as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

#[derive(Clone)]
struct Scheme {
    module: Arc<ModuleBox>,
    name: CString,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt: DigitLayout,
}

impl Scheme {
    pub fn new(handle: &Arc<Handle>, SchemeKey { dt }: SchemeKey) -> Result<Self, SchemeError> {
        let device = handle.device();
        let cc = device.compute_capability();
        let type_name = dt_name(dt);

        const CODE: &str = include_str!("add.cuh");
        let name = format!("add_{type_name}");
        let module = handle.compile_kernel(&name, cc, || {
            format!(
                r#"{CODE}
            
extern "C" __global__ void  add_{type_name}(
    {type_name} *__restrict__ c,
    {type_name}  const *__restrict__ a,
    {type_name} const *__restrict__ b,
    int const stride
){{
    add(c, a, b,stride);
}}"#
            )
        });

        Ok(Self {
            module,
            name: CString::new(name).unwrap(),
        })
    }
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
        rand::thread_rng().fill(&mut a[..]);
        rand::thread_rng().fill(&mut b[..]);
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
