use super::{args::Meta, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    get_static, infini::Device, Blob, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace,
};
use digit_layout::{types as ty, DigitLayout};
use infini_op::{infiniop, AsRaw, Descriptor};

pub struct Operator(Device);

impl Rope<Device> for Operator {
    fn build_sincos<QA>(
        dt: DigitLayout,
        nctx: usize,
        dh: usize,
        queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        fn generate_sin_cos_tables(max_seq_len: usize, dh: usize, theta: f32) -> Vec<[f32; 2]> {
            let len = max_seq_len * dh;
            let mut ans = vec![[0.; 2]; len];
            let (sin, cos) = ans.split_at_mut(len / 2);

            let half_dh = dh / 2;
            for i in 0..max_seq_len {
                for j in 0..half_dh {
                    let k = i * half_dh + j;
                    let (sin_, cos_) = (i as f32 / theta.powf(j as f32 / half_dh as f32)).sin_cos();
                    sin[k] = [sin_, sin_];
                    cos[k] = [cos_, cos_];
                }
            }
            ans
        }

        assert_eq!(dt, ty::F32);
        let host = generate_sin_cos_tables(nctx, dh, 1e4);
        let mut mem = queue_alloc.alloc(size_of_val(host.as_slice()));
        queue_alloc.queue().memcpy_h2d(&mut mem, &host);
        queue_alloc.queue().synchronize();
        SinCosTable { nctx, mem }
    }

    fn build_pos<I, QA>(dt: DigitLayout, nt: usize, iter: I, queue_alloc: &QA) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut host = Blob::new(dt.nbytes() * nt);
        match dt {
            ty::U32 => fill_pos(host.as_mut_ptr().cast::<u32>(), nt, iter),
            ty::U64 => fill_pos(host.as_mut_ptr().cast::<u64>(), nt, iter),
            _ => todo!(),
        }

        let mut blob = queue_alloc.alloc(host.len());
        queue_alloc.queue().memcpy_h2d(&mut blob, &host);
        queue_alloc.queue().synchronize();
        blob
    }
}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    #[inline]
    fn new(node: &Self::TopoNode) -> Self {
        Self(node.clone())
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
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt_t, dt_p, .. } = args.meta()?;
        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            sin_layout,
            cos_layout,
            ..
        } = args;

        let &[nctx, nh, dh] = t_layout.shape() else {
            unreachable!()
        };
        let &[ncs, nhs, dhs] = t_layout.strides() else {
            unreachable!()
        };
        let &[ps] = p_layout.strides() else {
            unreachable!()
        };
        let &[sns, sds] = sin_layout.strides() else {
            unreachable!()
        };
        let &[snc, sdc] = cos_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nctx nh dh
            ncs nhs dhs
            ps
            sns sds
            snc sdc
        }

        let t = infini_op::Tensor::new(dt_t, [nctx, nh, dh], [ncs, nhs, dhs]);
        let p = infini_op::Tensor::new(dt_p, [nctx], [ps]);
        let sin = infini_op::Tensor::new(sin_layout.dt(), [nctx, dh], [sns, sds]);
        let cos = infini_op::Tensor::new(cos_layout.dt(), [nctx, dh], [snc, sdc]);

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateRoPEDescriptor(
                    self.0.handle().as_raw(),
                    ptr,
                    t.as_raw(),
                    p.as_raw(),
                    sin.as_raw(),
                    cos.as_raw()
                ))
            },
            infini_op::bindings::infiniopDestroyRoPEDescriptor,
        );
        let mut workspace_size = 0;
        infiniop!(infiniopGetRoPEWorkspaceSize(
            descriptor.as_raw(),
            &mut workspace_size
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);

        infiniop!(infiniopRoPE(
            descriptor.as_raw(),
            workspace.as_mut_ptr().cast(),
            workspace_size,
            t_base.cast(),
            p_base.cast(),
            args.sin_base.cast(),
            args.cos_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Device, Operator};
    use crate::{rope::Rope, Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};
    use std::ptr::null;

    fn dyn_args<H: Hardware>(dt_t: DigitLayout, dt_p: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            t_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 3], &[dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new_dyn(dt_p, &[dyn_()], &[dyn_()]),
            p_base: null(),
            sin_layout: TensorLayout::new_dyn(ty::F32, &[dyn_(); 2], &[dyn_(); 2]),
            sin_base: null(),
            cos_layout: TensorLayout::new_dyn(ty::F32, &[dyn_(); 2], &[dyn_(); 2]),
            cos_base: null(),
            theta: 0.,
        }
    }

    fn args<H: Hardware>(
        dt_t: DigitLayout,
        dt_p: DigitLayout,
        nt: usize,
        nh: usize,
        dh: usize,
        theta: f32,
        t_base: *mut H::Byte,
        p_base: *const H::Byte,
        sin_base: *const H::Byte,
        cos_base: *const H::Byte,
    ) -> Args<H> {
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        Args {
            t_layout: TensorLayout::from_arr(
                dt_t,
                &ArrayLayout::<3>::new_contiguous(&[nt, nh, dh], BigEndian, dt_t.nbytes()).slice(
                    1,
                    4,
                    1,
                    nh - 8,
                ),
            ),
            t_base,
            p_layout: TensorLayout::new_contiguous(dt_p, &[nt]),
            p_base,
            sin_layout: TensorLayout::new_contiguous(ty::F32, &[nt, dh]),
            sin_base,
            cos_layout: TensorLayout::new_contiguous(ty::F32, &[nt, dh]),
            cos_base,
            theta,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            infini::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use half::f16;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let mut cpu_op = RefOp::new(&Cpu);
        let mut dev_op = Operator::new(&dev);

        cpu_op.scheme(&dyn_args(ty::F64, ty::U32), 0).unwrap();
        dev_op.scheme(&dyn_args(ty::F16, ty::U64), 0).unwrap();

        const NT: usize = 7;
        let nh = 32;
        let dh = 64;

        let mut t = vec![0.0f64; NT * nh * dh];
        rand::rng().fill(&mut t[..]);
        let p: [u32; NT] = [0, 1, 2, 3, 7, 8, 1];
        let nctx = *p.iter().max().unwrap() as usize + 1;

        let t_ans = {
            let stream = dev.stream();
            let mut t = cast_load(&t, f16::from_f64, &stream);
            let p = cast_load(&p, |x| x as u64, &stream);
            let sincos = Operator::build_sincos(ty::F32, nctx, dh, &stream);
            let (sin, cos) = sincos.mem.split_at(sincos.mem.len() / 2);

            dev_op
                .launch(
                    &args(
                        ty::F16,
                        ty::U64,
                        NT,
                        nh,
                        dh,
                        1e4,
                        t.as_mut_ptr().cast(),
                        p.as_ptr().cast(),
                        sin.as_ptr().cast(),
                        cos.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();

            let mut host = vec![f16::ZERO; NT * nh * dh];
            dev.synchronize();
            dev.memcpy_d2h(&mut host, &t);
            host
        };

        let mut t_ref = t;
        cpu_op
            .launch(
                &args(
                    ty::F64,
                    ty::U32,
                    NT,
                    nh,
                    dh,
                    1e4,
                    t_ref.as_mut_ptr().cast(),
                    p.as_ptr().cast(),
                    null(),
                    null(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        let diff = t_ref
            .into_par_iter()
            .zip(t_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(1.5 * f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
