use super::{args::Meta, Args, RmsNorm};
use crate::{
    nvidia_gpu::{dt_name, Handle as Gpu, Internal as Handle, ModuleBox},
    utils::{get_static, sizeof},
};
use common::{
    scheme_not_compatible, scheme_not_set, shape_not_support, strides_not_support, LaunchError,
    QueueOf, SchemeError,
};
use dev_mempool::cuda::Version;
use digit_layout::DigitLayout;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<(Scheme, Arc<ModuleBox>)>,
}

impl RmsNorm<Gpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), SchemeError> {
        let Meta {
            dt_w,
            dt_a,
            n: _,
            d,
        } = args.meta()?;

        #[allow(unreachable_code, clippy::diverging_sub_expression)]
        let Some(&d) = d.get_static() else {
            self.scheme = Some((Scheme::Common { dt_w, dt_a }, todo!()));
        };

        let device = self.handle.device();
        let cc = device.compute_capability();
        let max_num_threads_block = device.block_limit().max_threads;

        if d <= max_num_threads_block {
            self.padding_scheme(dt_w, dt_a, d, cc)
        } else {
            self.folding_scheme(dt_w, dt_a, d, cc, max_num_threads_block, device.warp_size())
        }
    }

    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
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
        let &[nsy, dsy] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, dsx] = x_layout.strides() else {
            unreachable!()
        };
        let &[dsw] = w_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   d
            nsy dsy
            nsx dsx
            dsw
        }

        let unit = sizeof(dt_a)? as isize;
        if dsy != unit || dsx != unit || dsw != sizeof(dt_w)? as isize {
            return Err(strides_not_support("").into());
        };

        let (name, m, block_dims) = match self.scheme.as_ref() {
            Some((s, m)) => {
                if !s.is_match(dt_w, dt_a, d) {
                    return Err(scheme_not_compatible(""));
                }
                match s {
                    Scheme::Common { .. } => todo!(),
                    Scheme::Padding { name, .. } => (name, m, d),
                    Scheme::Folding {
                        name, block_size, ..
                    } => (name, m, *block_size),
                }
            }
            None => return Err(scheme_not_set("")),
        };

        let nsy = (nsy / unit) as i32;
        let nsx = (nsx / unit) as i32;
        let params = dev_mempool::cuda::params![y_base, nsy, x_base, nsx, w_base, epsilon];

        m.launch(name, n as u32, block_dims as u32, params.as_ptr(), 0, queue);
        Ok(())
    }
}

enum Scheme {
    Common {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
    },
    Padding {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        name: CString,
    },
    Folding {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        block_size: usize,
        name: CString,
    },
}

impl Scheme {
    fn is_match(&self, dt_w_: DigitLayout, dt_a_: DigitLayout, d_: usize) -> bool {
        match *self {
            Scheme::Common { dt_w, dt_a } => dt_w == dt_w_ && dt_a == dt_a_,
            Scheme::Padding { dt_w, dt_a, d, .. } => dt_w == dt_w_ && dt_a == dt_a_ && d == d_,
            Scheme::Folding { dt_w, dt_a, d, .. } => dt_w == dt_w_ && dt_a == dt_a_ && d == d_,
        }
    }
}

const CODE: &str = include_str!("rms_norm.cuh");

impl Operator {
    fn padding_scheme(
        &mut self,
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        cc: Version,
    ) -> Result<(), SchemeError> {
        let ww = sizeof(dt_w)? * 8;
        let wa = sizeof(dt_a)? * 8;
        let name = format!("rms_norm_padding_w{ww}a{wa}_{d}");
        let tw = dt_name(dt_w);
        let ta = dt_name(dt_a);
        let module = self.handle.compile_kernel(&name, cc, || {
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
        self.scheme = Some((
            Scheme::Padding {
                dt_w,
                dt_a,
                d,
                name: CString::new(name).unwrap(),
            },
            module,
        ));
        Ok(())
    }

    fn folding_scheme(
        &mut self,
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        cc: Version,
        max_num_threads_block: usize,
        num_threads_warp: usize,
    ) -> Result<(), SchemeError> {
        if d % num_threads_warp != 0 {
            Err(shape_not_support(format!(
                "normalization shape {d} must be multiple of warp size {num_threads_warp}"
            )))?
        }
        let max_num_warp_block = max_num_threads_block / num_threads_warp;
        // num_warp_block in [1, max_num_warp_block]
        // num_threads_warp
        // num_items_thread in [1, 2, 4, 8] // 8 = 128bit / sizeof(half)
        // TODO 也许还能分得更好
        let to_divid = d / num_threads_warp;
        let num_warps_block = max_num_warp_block;
        let num_threads_block = num_threads_warp * num_warps_block;
        let num_items_thread = (to_divid + num_warps_block - 1) / num_warps_block;

        let ww = sizeof(dt_w)? * 8;
        let wa = sizeof(dt_a)? * 8;
        let name = format!("rms_norm_padding_w{ww}a{wa}_{num_threads_block}x{num_items_thread}");
        let tw = dt_name(dt_w);
        let ta = dt_name(dt_a);

        let module = self.handle.compile_kernel(&name, cc, || {
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

        self.scheme = Some((
            Scheme::Folding {
                dt_w,
                dt_a,
                d,
                block_size: num_threads_block,
                name: CString::new(name).unwrap(),
            },
            module,
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator, Scheme};
    use common::{Handle, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F32, F64},
        DigitLayout,
    };

    fn dyn_args<H: Handle>(dt_w: DigitLayout, dt_a: DigitLayout, d: usize) -> Args<H> {
        use common::dyn_;
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

    fn args<H: Handle>(
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
            op.scheme(&dyn_args(F32, F16, d)).unwrap();
            let (scheme, module) = op.scheme.as_ref().unwrap();
            match scheme {
                Scheme::Common { .. } => todo!(),
                Scheme::Padding { name, .. } => gpu.apply(|ctx| {
                    println!(
                        "{}\n{}",
                        name.to_str().unwrap(),
                        module.load(name, ctx).info()
                    );
                }),
                Scheme::Folding { name, .. } => gpu.apply(|ctx| {
                    println!(
                        "{}\n{}",
                        name.to_str().unwrap(),
                        module.load(name, ctx).info()
                    );
                }),
            }
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Handle as Cpu, ThisThread},
            nvidia_gpu::cast_load,
            utils::{Diff, ErrorCollector},
        };
        use dev_mempool::cuda::memcpy_d2h;
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
            cpu_op.scheme(&dyn_args(F64, F64, d)).unwrap();
            gpu_op.scheme(&dyn_args(F32, F16, d)).unwrap();

            let mut x = vec![0.0f64; n * d];
            let mut w = vec![0.0f64; d];
            rand::thread_rng().fill(&mut x[..]);
            rand::thread_rng().fill(&mut w[..]);
            let x = x;
            let w = w;

            let y_ans = gpu.apply(|ctx| {
                let stream = ctx.stream();
                let mut y = stream.malloc::<f16>(n * d);
                let x = cast_load(&x, |&x| f16::from_f64(x), &stream);
                let w = cast_load(&w, |&x| x as f32, &stream);
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
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; n * d];
                memcpy_d2h(&mut host, &y);
                host
            });

            let mut y_ref = vec![0.0; n * d];
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
