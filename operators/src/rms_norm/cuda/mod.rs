use super::{args::Meta, Args, RmsNorm};
use crate::{
    cuda::{dt_name, Gpu, Handle, ModuleBox},
    get_static, shape_not_support, strides_not_support, ByteOf, LaunchError, QueueAlloc,
    SchemeDiversity, SchemeError,
};
use digit_layout::DigitLayout;
use lru::LruCache;
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

pub struct Operator {
    handle: Arc<Handle>,
    schemes: Mutex<LruCache<SchemeKey, Scheme>>,
}

impl RmsNorm<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
            schemes: node.0.scheme_cache(SchemeDiversity::Low),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt_a, dt_w, d, .. } = args.meta()?;
        get_static!(d);

        let key = SchemeKey { dt_a, dt_w, d };
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
        let Meta { dt_w, dt_a, n, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            w_layout,
            w_base,
            epsilon,
        } = args;
        let &[yns, yds] = y_layout.strides() else {
            unreachable!()
        };
        let &[xns, xds] = x_layout.strides() else {
            unreachable!()
        };
        let &[wds] = w_layout.strides() else {
            unreachable!()
        };

        get_static! {
             n   d
            yns yds
            xns xds
                wds
        }

        let unit = dt_a.nbytes() as isize;
        if yds != unit || xds != unit || wds != dt_w.nbytes() as isize {
            return Err(strides_not_support("").into());
        };

        let key = SchemeKey { dt_a, dt_w, d };
        let scheme = self
            .schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?
            .clone();

        let nsy = (yns / unit) as i32;
        let nsx = (xns / unit) as i32;
        let params = cuda::params![y_base, nsy, x_base, nsx, w_base, epsilon];

        scheme.module.launch(
            &scheme.name,
            n as u32,
            match scheme.ty {
                SchemeType::Padding => d,
                SchemeType::Folding { block_size } => block_size,
            } as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt_a: DigitLayout,
    dt_w: DigitLayout,
    d: usize,
}

#[derive(Clone)]
struct Scheme {
    ty: SchemeType,
    module: Arc<ModuleBox>,
    name: CString,
}

#[derive(Clone)]
enum SchemeType {
    Padding,
    Folding { block_size: usize },
}

impl Scheme {
    pub fn new(
        handle: &Arc<Handle>,
        SchemeKey { dt_a, dt_w, d }: SchemeKey,
    ) -> Result<Self, SchemeError> {
        let device = handle.device();
        let cc = device.compute_capability();
        let block_size = device.block_limit().max_threads;

        let ta = dt_name(dt_a);
        let tw = dt_name(dt_w);

        const CODE: &str = include_str!("rms_norm.cuh");
        if d <= block_size {
            let name = format!("rms_norm_{ta}_{tw}_padding_{d}");
            let module = handle.compile_kernel(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    padding<{d}>
    (y, stride_y, x, stride_x, w, epsilon);
}}"#
                )
            });

            Ok(Self {
                ty: SchemeType::Padding,
                module,
                name: CString::new(name).unwrap(),
            })
        } else {
            let n_threads_warp = device.warp_size();
            if d % n_threads_warp != 0 {
                Err(shape_not_support(format!(
                    "normalization shape {d} must be multiple of warp size {n_threads_warp}"
                )))?
            }
            let max_num_warp_block = block_size / n_threads_warp;
            // num_warp_block in [1, max_num_warp_block]
            // num_threads_warp
            // num_items_thread in [1, 2, 4, 8] // 8 = 128bit / sizeof(half)
            // TODO 也许还能分得更好
            let to_divid = d / n_threads_warp;
            let num_warps_block = max_num_warp_block;
            let num_threads_block = n_threads_warp * num_warps_block;
            let num_items_thread = to_divid.div_ceil(num_warps_block);

            let name = format!("rms_norm_{ta}_{tw}_folding_{num_threads_block}x{num_items_thread}");
            let module = handle.compile_kernel(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    folding<{num_threads_block}, {num_items_thread}>
    (y, stride_y, x, stride_x, w, epsilon, {d});
}}"#
                )
            });

            Ok(Self {
                ty: SchemeType::Folding { block_size },
                module,
                name: CString::new(name).unwrap(),
            })
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F32, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt_w: DigitLayout, dt_a: DigitLayout, d: usize) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            y_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            y_base: null_mut(),
            x_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            x_base: null(),
            w_layout: TensorLayout::new_dyn(dt_w, &[d.into()], &[dyn_()]),
            w_base: null(),
            epsilon: 1e-5,
        }
    }

    fn args<H: Hardware>(
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        n: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        w_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt_a, &[n, d]);
        Args {
            y_layout: layout.clone(),
            y_base,
            x_layout: layout,
            x_base,
            w_layout: TensorLayout::new_contiguous(dt_w, &[d]),
            w_base,
            epsilon: 1e-5,
        }
    }

    #[test]
    fn test_compile() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        for k in 8..=13 {
            let d = 1 << k;
            op.scheme(&dyn_args(F32, F16, d), 0).unwrap();
            let scheme = op.schemes.lock().unwrap().iter().next().unwrap().1.clone();
            gpu.apply(|ctx| {
                println!(
                    "{}\n{}",
                    scheme.name.to_str().unwrap(),
                    scheme.module.load(&scheme.name, ctx).info()
                )
            });
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
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        for k in 8..=13 {
            let n = 4;
            let d = 1 << k;
            cpu_op.scheme(&dyn_args(F64, F64, d), 0).unwrap();
            gpu_op.scheme(&dyn_args(F32, F16, d), 0).unwrap();

            let mut x = vec![0.0f64; n * d];
            let mut w = vec![0.0f64; d];
            rand::rng().fill(&mut x[..]);
            rand::rng().fill(&mut w[..]);
            let x = x;
            let w = w;

            let y_ans = gpu.apply(|ctx| {
                let stream = ctx.stream();
                #[cfg(use_nvidia)]
                let rt = &stream;
                #[cfg(use_iluvatar)]
                let rt = ctx;
                let mut y = rt.malloc::<f16>(n * d);
                let x = cast_load(&x, f16::from_f64, &stream);
                let w = cast_load(&w, |x| x as f32, &stream);
                gpu_op
                    .launch(
                        &args(
                            F32,
                            F16,
                            n,
                            d,
                            y.as_mut_ptr().cast(),
                            x.as_ptr().cast(),
                            w.as_ptr().cast(),
                        ),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; n * d];
                memcpy_d2h(&mut host, &y);
                host
            });

            let mut y_ref = vec![0.; n * d];
            cpu_op
                .launch(
                    &args(
                        F64,
                        F64,
                        n,
                        d,
                        y_ref.as_mut_ptr().cast(),
                        x.as_ptr().cast(),
                        w.as_ptr().cast(),
                    ),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = y_ref
                .into_par_iter()
                .zip(y_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 1e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
