impl_op!(nvidia_gpu, Gpu);

#[cfg(test)]
mod test {
    use super::{super::Args, Operator};
    use crate::{nvidia_gpu::Gpu, Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout, nt: usize, di: usize, d: usize) -> Args<H> {
        use crate::dyn_;
        Args::new_null(
            TensorLayout::new_dyn(dt, &[nt.into(), d.into()], &[dyn_(); 2]),
            TensorLayout::new_dyn(dt, &[nt.into(), d.into()], &[dyn_(); 2]),
            TensorLayout::new_dyn(dt, &[d.into(), (di * 2).into()], &[dyn_(); 2]),
            TensorLayout::new_dyn(dt, &[di.into(), d.into()], &[dyn_(); 2]),
            1.,
            true,
        )
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nt: usize,
        di: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        w_gate_up_base: *const H::Byte,
        w_down_base: *const H::Byte,
        down_alpha: f32,
        residual: bool,
    ) -> Args<H> {
        Args {
            y_layout: TensorLayout::new_contiguous(dt, &[nt, d]),
            y_base,
            x_layout: TensorLayout::new_contiguous(dt, &[nt, d]),
            x_base,
            w_gate_up_layout: TensorLayout::new_contiguous(dt, &[d, di * 2]),
            w_gate_up_base,
            w_down_layout: TensorLayout::new_contiguous(dt, &[di, d]),
            w_down_base,
            down_alpha,
            residual,
        }
    }

    #[test]
    fn test_compile() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        let workspace = op.scheme(&dyn_args(ty::F16, 14, 5632, 2048), usize::MAX);
        println!("workspace: {workspace:?}");
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            nvidia_gpu::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use dev_mempool::cuda::memcpy_d2h;
        use half::f16;
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let nt = 1;
        let di = 11 * 512;
        let d = 4 * 512;

        let cpu_op = RefOp::new(&Cpu);
        let gpu_op = Operator::new(&gpu);

        let mut y = vec![0.0f64; nt * d];
        let mut x = vec![0.0f64; nt * d];
        let mut w_gate_up = vec![0.0f64; d * di * 2];
        let mut w_down = vec![0.0f64; di * d];
        rand::thread_rng().fill(&mut y[..]);
        rand::thread_rng().fill(&mut x[..]);
        rand::thread_rng().fill(&mut w_gate_up[..]);
        rand::thread_rng().fill(&mut w_down[..]);
        x.iter_mut().for_each(|x| *x *= 1e-4);
        let y = y;
        let x = x;
        let w_gate_up = w_gate_up;
        let w_down = w_down;

        let y_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut y = cast_load(&y, f16::from_f64, &stream);
            let x = cast_load(&x, f16::from_f64, &stream);
            let w_gate_up = cast_load(&w_gate_up, f16::from_f64, &stream);
            let w_down = cast_load(&w_down, f16::from_f64, &stream);
            gpu_op
                .launch(
                    &args(
                        ty::F16,
                        nt,
                        di,
                        d,
                        y.as_mut_ptr().cast(),
                        x.as_ptr().cast(),
                        w_gate_up.as_ptr().cast(),
                        w_down.as_ptr().cast(),
                        1.0,
                        true,
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; nt * d];
            memcpy_d2h(&mut host, &y);
            host
        });

        let mut y_ref = y;
        cpu_op
            .launch(
                &args(
                    ty::F64,
                    nt,
                    di,
                    d,
                    y_ref.as_mut_ptr().cast(),
                    x.as_ptr().cast(),
                    w_gate_up.as_ptr().cast(),
                    w_down.as_ptr().cast(),
                    1.0,
                    true,
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = y_ref
            .into_iter()
            .zip(y_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 3e-3);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
