impl_op!(cuda, Gpu);

#[cfg(test)]
mod test {
    use super::{super::Args, Operator};
    use crate::{cuda::Gpu, ByteOf, Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        seq: usize,
        dh: usize,
        pos: usize,
    ) -> Args<H> {
        use crate::dyn_;
        Args::new_null(
            TensorLayout::new_dyn(dt, &[nh.into(), seq.into(), dh.into()], &[dyn_(); 3]),
            TensorLayout::new_dyn(dt, &[dyn_(), seq.into(), dh.into()], &[dyn_(); 3]),
            TensorLayout::new_dyn(dt, &[dyn_(), seq.into(), dh.into()], &[dyn_(); 3]),
            TensorLayout::new_dyn(dt, &[nh.into(), seq.into(), dh.into()], &[dyn_(); 3]),
            TensorLayout::new_dyn(dt, &[nh.into(), seq.into(), dh.into()], &[dyn_(); 3]),
            TensorLayout::new_dyn(dt, &[nh.into(), seq.into(), dh.into()], &[dyn_(); 3]),
            crate::fuesd_softmax::AttnMask::Causal,
            pos.into(),
        )
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        nkvh: usize,
        seq: usize,
        dh: usize,
        pos: usize,
        q_base: *mut ByteOf<H>,
        k_base: *const ByteOf<H>,
        v_base: *const ByteOf<H>,
        o_base: *mut ByteOf<H>,
        k_cache_base: *mut ByteOf<H>,
        v_cache_base: *mut ByteOf<H>,
    ) -> Args<H> {
        Args {
            q_layout: TensorLayout::new_contiguous(dt, &[nh, seq, dh]),
            k_layout: TensorLayout::new_contiguous(dt, &[nkvh, seq, dh]),
            v_layout: TensorLayout::new_contiguous(dt, &[nkvh, seq, dh]),
            o_layout: TensorLayout::new_contiguous(dt, &[nh, seq, dh]),
            k_cache_layout: TensorLayout::new_contiguous(dt, &[nkvh, seq + pos, dh]),
            v_cache_layout: TensorLayout::new_contiguous(dt, &[nkvh, seq + pos, dh]),
            q_base,
            k_base,
            v_base,
            o_base,
            k_cache_base,
            v_cache_base,
            mask: crate::fuesd_softmax::AttnMask::Causal,
            pos: pos.into(),
        }
    }

    #[test]
    fn test_compile() {
        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        let workspace = op.scheme(&dyn_args(ty::F16, 32, 7, 64, 13), usize::MAX);
        println!("workspace: {workspace:?}");
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

        let nh = 32;
        let nkvh = 4;
        let seq = 7;
        let dh = 64;
        let pos = 13;

        let cpu_op = RefOp::new(&Cpu);
        let gpu_op = Operator::new(&gpu);

        let mut q = vec![0.0f64; nh * seq * dh];
        let mut k = vec![0.0f64; nkvh * seq * dh];
        let mut v = vec![0.0f64; nkvh * seq * dh];
        let o = vec![0.0f64; nh * seq * dh];
        let mut k_cache = vec![0.0f64; nkvh * (pos + seq) * dh];
        let mut v_cache = vec![0.0f64; nkvh * (pos + seq) * dh];
        rand::rng().fill(&mut q[..]);
        rand::rng().fill(&mut k[..]);
        rand::rng().fill(&mut v[..]);
        rand::rng().fill(&mut k_cache[..]);
        rand::rng().fill(&mut v_cache[..]);
        let k = k;
        let v = v;

        let o_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let mut q = cast_load(&q, f16::from_f64, &stream);
            let k = cast_load(&k, f16::from_f64, &stream);
            let v = cast_load(&v, f16::from_f64, &stream);
            let mut o = rt.malloc::<f16>(o.len());
            let mut k_cache = cast_load(&k_cache, f16::from_f64, &stream);
            let mut v_cache = cast_load(&v_cache, f16::from_f64, &stream);
            gpu_op
                .launch(
                    &args(
                        ty::F16,
                        nh,
                        nkvh,
                        seq,
                        dh,
                        pos,
                        q.as_mut_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        o.as_mut_ptr(),
                        k_cache.as_mut_ptr(),
                        v_cache.as_mut_ptr(),
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
                    dh,
                    pos,
                    q.as_mut_ptr().cast(),
                    k.as_ptr().cast(),
                    v.as_ptr().cast(),
                    o_ref.as_mut_ptr().cast(),
                    k_cache.as_mut_ptr().cast(),
                    v_cache.as_mut_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = o_ref
            .into_iter()
            .zip(o_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 1e-3);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
