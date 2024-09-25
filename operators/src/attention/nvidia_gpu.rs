﻿impl_op!(nvidia_gpu, Gpu);

#[cfg(test)]
mod test {
    use super::{
        super::{args::Meta, Args},
        Operator,
    };
    use crate::{nvidia_gpu::Gpu, ByteOf, Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout, nh: usize, seq: usize, att: usize) -> Args<H> {
        use crate::dyn_;
        Meta {
            dt,
            nh: nh.into(),
            nkvh: dyn_(),
            seq: seq.into(),
            att: att.into(),
            dh: dyn_(),
        }
        .into()
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        nkvh: usize,
        seq: usize,
        att: usize,
        dh: usize,
        q_base: *mut ByteOf<H>,
        k_base: *const ByteOf<H>,
        v_base: *const ByteOf<H>,
        o_base: *mut ByteOf<H>,
    ) -> Args<H> {
        Args {
            q_layout: TensorLayout::new_contiguous(dt, &[nh, seq, dh]),
            k_layout: TensorLayout::new_contiguous(dt, &[nkvh, att, dh]),
            v_layout: TensorLayout::new_contiguous(dt, &[nkvh, att, dh]),
            o_layout: TensorLayout::new_contiguous(dt, &[nh, seq, dh]),
            q_base,
            k_base,
            v_base,
            o_base,
        }
    }

    #[test]
    fn test_compile() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        let workspace = op.scheme(&dyn_args(ty::F16, 32, 7, 127), usize::MAX);
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
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let nh = 32;
        let nkvh = 4;
        let seq = 7;
        let att = 127;
        let dh = 64;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        println!(
            "cpu workspace: {}",
            cpu_op
                .scheme(&dyn_args(ty::F64, nh, seq, att), usize::MAX)
                .unwrap()
        );
        println!(
            "gpu workspace: {}",
            gpu_op
                .scheme(&dyn_args(ty::F16, nh, seq, att), usize::MAX)
                .unwrap()
        );

        let mut q = vec![0.0f64; nh * seq * dh];
        let mut k = vec![0.0f64; nkvh * att * dh];
        let mut v = vec![0.0f64; nkvh * att * dh];
        let o = vec![0.0f64; nh * seq * dh];
        rand::thread_rng().fill(&mut q[..]);
        rand::thread_rng().fill(&mut k[..]);
        rand::thread_rng().fill(&mut v[..]);
        let k = k;
        let v = v;

        let o_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut q = cast_load(&q, f16::from_f64, &stream);
            let k = cast_load(&k, f16::from_f64, &stream);
            let v = cast_load(&v, f16::from_f64, &stream);
            let mut o = stream.malloc::<f16>(o.len());
            gpu_op
                .launch(
                    &args(
                        ty::F16,
                        nh,
                        nkvh,
                        seq,
                        att,
                        dh,
                        q.as_mut_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        o.as_mut_ptr(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();

            let mut host = vec![f16::ZERO; nh * seq * dh];
            memcpy_d2h(&mut host, &o);
            host
        });

        let mut o_ref = o;
        cpu_op
            .launch(
                &args(
                    ty::F64,
                    nh,
                    nkvh,
                    seq,
                    att,
                    dh,
                    q.as_mut_ptr().cast(),
                    k.as_ptr().cast(),
                    v.as_ptr().cast(),
                    o_ref.as_mut_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = o_ref
            .into_par_iter()
            .zip(o_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 3e-3);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
