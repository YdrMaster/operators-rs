use super::{args::Meta, Args, Rope};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{algebraic, locate_error, ErrorPosition, QueueOf};
use digit_layout::types::{F16, U32};
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    max_threads_block: usize,
    scheme: Option<Arc<ModuleBox>>,
}
const NAME: &str = "rope_f16";
const CODE: &str = include_str!("rope.cuh");

impl Rope<Gpu> for Operator {}

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
        let Meta { dt_t, dt_p, .. } = args.meta()?;

        if dt_t != F16 || dt_p != U32 {
            todo!()
        }

        let cc = self.handle.device().compute_capability();
        self.scheme = Some(self.handle.compile_kernel(NAME, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {NAME}(
    half2 *__restrict__ t,
    int const stride_token,
    int const stride_head,
    unsigned int const *__restrict__ pos,
    float theta
){{
    padding(t, stride_token, stride_head, pos, theta);
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
        let Meta {
            dt_t, dt_p, nt, dh, ..
        } = args.meta()?;

        if dt_t != F16 || dt_p != U32 {
            todo!()
        }

        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            ..
        } = args;
        let &[_, nh, _] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        get_or_err!(nt);
        get_or_err!(nh);
        get_or_err!(dh);
        get_or_err!(st);
        get_or_err!(sh);
        get_or_err!(sd);
        get_or_err!(sp);

        let unit = algebraic!(dt_t)? as isize;
        if sd != unit || sp != size_of::<u32>() as isize {
            return Err(locate_error!("Unsupported layout"));
        };

        let Some(m) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };

        let dh = dh / 2;
        let st = (st / unit / 2) as i32;
        let sh = (sh / unit / 2) as i32;
        let params = cuda::params![t_base, st, sh, p_base, theta];

        if self.max_threads_block % dh != 0 {
            return Err(locate_error!());
        }

        let max_nh_l = (self.max_threads_block / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;

        m.launch(
            CString::new(NAME).unwrap(),
            (nt as _, nh_h as _),
            (nh_l as _, dh as _),
            params.as_ptr(),
            0,
            queue,
        );
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use common::{Handle, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64, U32},
        DigitLayout,
    };

    fn dyn_args<H: Handle>(dt_t: DigitLayout, dt_p: DigitLayout) -> Args<H> {
        use common::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            t_layout: TensorLayout::new(dt_t, &[dyn_(); 3], &[dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new(dt_p, &[dyn_()], &[dyn_()]),
            p_base: null(),
            sin_layout: TensorLayout::new(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            sin_base: null(),
            cos_layout: TensorLayout::new(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            cos_base: null(),
            theta: 0.,
        }
    }

    fn args<H: Handle>(
        dt_t: DigitLayout,
        dt_p: DigitLayout,
        nt: usize,
        nh: usize,
        dh: usize,
        theta: f32,
        t_base: *mut H::Byte,
        p_base: *const H::Byte,
    ) -> Args<H> {
        use std::ptr::null;
        Args {
            t_layout: TensorLayout::new_contiguous(dt_t, [nt, nh, dh]),
            t_base,
            p_layout: TensorLayout::new_contiguous(dt_p, [nt]),
            p_base,
            sin_layout: TensorLayout::new_contiguous(dt_t, [0, dh]),
            sin_base: null(),
            cos_layout: TensorLayout::new_contiguous(dt_t, [0, dh]),
            cos_base: null(),
            theta,
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
        op.scheme(&dyn_args(F16, U32)).unwrap();

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
        cpu_op.scheme(&dyn_args(F64, U32)).unwrap();
        gpu_op.scheme(&dyn_args(F16, U32)).unwrap();

        const NT: usize = 7;
        let nh = 32;
        let dh = 64;

        let mut t = vec![0.0f64; NT * nh * dh];
        rand::thread_rng().fill(&mut t[..]);
        let p: [u32; NT] = [0, 1, 2, 3, 7, 8, 1];

        let t_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut t = cast_load(&t, |&x| f16::from_f64(x), &stream);
            let p = stream.from_host(&p);
            gpu_op
                .launch(
                    &args(
                        F16,
                        U32,
                        NT,
                        nh,
                        dh,
                        1e4,
                        t.as_mut_ptr().cast(),
                        p.as_ptr().cast(),
                    ),
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; NT * nh * dh];
            memcpy_d2h(&mut host, &t);
            host
        });

        let mut t_ref = t;
        cpu_op
            .launch(
                &args(
                    F64,
                    U32,
                    NT,
                    nh,
                    dh,
                    1e4,
                    t_ref.as_mut_ptr().cast(),
                    p.as_ptr().cast(),
                ),
                &ThisThread,
            )
            .unwrap();

        let diff = t_ref
            .into_par_iter()
            .zip(t_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
