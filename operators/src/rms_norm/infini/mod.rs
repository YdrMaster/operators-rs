use super::{args::Meta, Args, RmsNorm};
use crate::{get_static, infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace};
use infini_op::{infiniop, AsRaw, Descriptor};

pub struct Operator(Device);

impl RmsNorm<Device> for Operator {}

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

        let y = infini_op::Tensor::new(dt_a, [n, d], [yns, yds]);
        let x = infini_op::Tensor::new(dt_a, [n, d], [xns, xds]);
        let w = infini_op::Tensor::new(dt_w, [d], [wds]);

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateRMSNormDescriptor(
                    self.0.handle().as_raw(),
                    ptr,
                    y.as_raw(),
                    x.as_raw(),
                    w.as_raw(),
                    *epsilon,
                ))
            },
            infini_op::bindings::infiniopDestroyRMSNormDescriptor,
        );
        let mut workspace_size = 0;
        infiniop!(infiniopGetRMSNormWorkspaceSize(
            descriptor.as_raw(),
            &mut workspace_size
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);
        infiniop!(infiniopRMSNorm(
            descriptor.as_raw(),
            workspace.as_mut_ptr().cast(),
            workspace_size,
            y_base.cast(),
            x_base.cast(),
            w_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Device, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F32, F64},
        DigitLayout,
    };
    use rayon::iter::ParallelIterator;

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
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            infini::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use half::f16;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let mut cpu_op = RefOp::new(&Cpu);
        let mut dev_op = Operator::new(&dev);

        for k in 8..=13 {
            let n = 4;
            let d = 1 << k;
            cpu_op.scheme(&dyn_args(F64, F64, d), 0).unwrap();
            dev_op.scheme(&dyn_args(F32, F16, d), 0).unwrap();

            let mut x = vec![0.0f64; n * d];
            let mut w = vec![0.0f64; d];
            rand::rng().fill(&mut x[..]);
            rand::rng().fill(&mut w[..]);
            let x = x;
            let w = w;

            let y_ans = {
                let stream = dev.stream();
                let mut y = stream.malloc::<f16>(n * d);
                let x = cast_load(&x, f16::from_f64, &stream);
                let w = cast_load(&w, |x| x as f32, &stream);
                dev_op
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
                dev.memcpy_d2h(&mut host, &y);
                host
            };

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
