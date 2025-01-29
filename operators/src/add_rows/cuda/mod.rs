use super::{AddRows, Args};
use crate::{
    add_rows::args::Meta,
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

impl AddRows<Gpu> for Operator {}

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
        let Meta { batch: b, n, m, .. } = args.meta()?;

        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
            idx_layout,
            idx_base,
        } = args;
        let &[bsd, msd, nsd] = dst_layout.strides() else {
            unreachable!()
        };
        let &[kss, nss] = src_layout.strides() else {
            unreachable!()
        };
        let &[bsi, msi] = idx_layout.strides() else {
            unreachable!()
        };

        get_static! {
            b   n   m
            bsd msd nsd
            bsi msi nss kss
        }
        let unit_dst = dst_layout.dt().nbytes() as isize;
        let unit_idx = idx_layout.dt().nbytes() as isize;
        if nsd != unit_dst || nss != unit_dst || msi != unit_idx {
            return Err(strides_not_support("").into());
        };
        fn cast(strides: &[isize], size: usize) -> Vec<isize> {
            strides.iter().map(|x| x / size as isize).collect()
        }
        let &[bsd, msd, kss] = cast(&[bsd, msd, kss], unit_dst as usize).as_slice() else {
            todo!()
        };
        let &[bsi] = cast(&[bsi], unit_idx as usize).as_slice() else {
            todo!()
        };
        let params = cuda::params![dst_base, src_base, idx_base, bsd, msd, kss, bsi];
        let block = gcd(self.max_threads_block, n);
        let dimx = n.div_ceil(block);
        let key = SchemeKey {
            dt: dst_layout.dt(),
        };
        let scheme = self
            .schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?
            .clone();
        scheme.module.launch(
            &scheme.name,
            (b as _, m as _, dimx as _),
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

        const CODE: &str = include_str!("add_rows.cuh");
        let name = format!("add_rows_{type_name}");
        let module = handle.compile_kernel(&name, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {name}(
    {type_name} *__restrict__ dst,
    {type_name} const *__restrict__ src,
    unsigned int const *__restrict__ idx,
    int const stride_d_b,
    int const stride_d_m,
    int const stride_s,
    int const stride_i
){{
    add_rows(dst, src, idx,stride_d_b, stride_d_m, stride_s, stride_i);
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
    use crate::{cuda::cast_load, dyn_, Hardware, Operator as _, TensorLayout};
    use cuda::memcpy_d2h;
    use digit_layout::{
        types::{F16, F64, U32},
        DigitLayout,
    };
    use half::f16;
    use std::ptr::null;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::null_mut;
        Args {
            dst_layout: TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
            idx_layout: TensorLayout::new_dyn(U32, &[dyn_(); 2], &[dyn_(); 2]),
            idx_base: null(),
        }
    }
    fn args<H: Hardware>(
        dt: DigitLayout,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        d_base: *mut H::Byte,
        s_base: *const H::Byte,
        i_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new_contiguous(dt, &[b, m, n]),
            dst_base: d_base,
            src_layout: TensorLayout::new_contiguous(dt, &[k, n]),
            src_base: s_base,
            idx_layout: TensorLayout::new_contiguous(U32, &[b, m]),
            idx_base: i_base,
        }
    }
    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            test_utils::{Diff, ErrorCollector},
        };
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16), 0).unwrap();

        let b = 1;
        let m = 10;
        let n = 2048;
        let k = m;
        let mut d = vec![0.1f64; b * m * n];
        let mut s = vec![0.1f64; k * n];
        let i: Vec<u32> = (0..=m).cycle().take(m * b).map(|x| x as u32).collect(); // 收集结果到 Vec 中
        rand::rng().fill(&mut d[..]);
        rand::rng().fill(&mut s[..]);
        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut d = cast_load(&d, f16::from_f64, &stream);
            let s = cast_load(&s, f16::from_f64, &stream);
            let i = cast_load(&i, u32::from, &stream);
            gpu_op
                .launch(
                    &args(
                        F16,
                        b,
                        m,
                        n,
                        k,
                        d.as_mut_ptr().cast(),
                        s.as_ptr().cast(),
                        i.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; b * m * n];
            memcpy_d2h(&mut host, &d);
            host
        });
        cpu_op
            .launch(
                &args(
                    F64,
                    b,
                    m,
                    n,
                    k,
                    d.as_mut_ptr().cast(),
                    s.as_ptr().cast(),
                    i.as_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        let diff = d
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
