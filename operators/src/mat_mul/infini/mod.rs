use super::{Args, MatMul};
use crate::{
    infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError, TensorLayout, Workspace,
};
use infini_op::{infiniop, AsRaw, Descriptor};

pub struct Operator(Device);

impl MatMul<Device> for Operator {}

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
        let Args {
            c_layout,
            c_base,
            beta,
            a_layout,
            a_base,
            b_layout,
            b_base,
            alpha,
        } = args;

        fn tensor(layout: &TensorLayout) -> infini_op::Tensor {
            infini_op::Tensor::new(
                layout.dt(),
                layout.shape().iter().map(|&x| *x.get_static().unwrap()),
                layout.strides().iter().map(|&x| *x.get_static().unwrap()),
            )
        }

        let c = tensor(c_layout);
        let a = tensor(a_layout);
        let b = tensor(b_layout);

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateMatmulDescriptor(
                    self.0.handle().as_raw(),
                    ptr,
                    c.as_raw(),
                    *alpha,
                    a.as_raw(),
                    b.as_raw(),
                    *beta,
                ))
            },
            infini_op::bindings::infiniopDestroyMatmulDescriptor,
        );
        let mut workspace_size = 0;
        infiniop!(infiniopGetMatmulWorkspaceSize(
            descriptor.as_raw(),
            &mut workspace_size
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);
        infiniop!(infiniopMatmul(
            descriptor.as_raw(),
            workspace.as_mut_ptr().cast(),
            workspace_size,
            c_base.cast(),
            a_base.cast(),
            b_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        if self.0.device_type() == infini_rt::DeviceType::DEVICE_ASCEND {
            queue_alloc.queue().synchronize();
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    const ALPHA: f32 = 0.5;
    const BETA: f32 = 1.;

    fn args<H: Hardware>(
        dt: DigitLayout,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[batch, m, n]),
            c_base,
            beta: BETA,
            a_layout: TensorLayout::new_contiguous(dt, &[batch, m, k]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[batch, k, n]),
            b_base,
            alpha: ALPHA,
        }
    }

    #[test]
    fn test_compute() {
        use super::{super::common_cpu::Operator as RefOp, Device, Operator};
        use crate::{
            common_cpu::{Cpu, ThisThread},
            infini::cast_load,
            test_utils::{Diff, ErrorCollector},
            Operator as _,
        };
        use digit_layout::types::{F16, F64};
        use half::f16;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let batch = 4;
        let k = 128;
        let n = 352;

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let cpu_op = RefOp::new(&Cpu);
        let dev_op = Operator::new(&dev);

        for m in [1, 7, 64, 128] {
            let mut a = vec![0.0f64; batch * m * k];
            let mut b = vec![0.0f64; batch * k * n];
            let mut c = vec![0.0f64; batch * m * n];
            rand::rng().fill(&mut a[..]);
            rand::rng().fill(&mut b[..]);
            rand::rng().fill(&mut c[..]);
            let a = a;
            let b = b;

            let c_ans = {
                let stream = dev.stream();
                let mut c = cast_load(&c, f16::from_f64, &stream);
                let a = cast_load(&a, f16::from_f64, &stream);
                let b = cast_load(&b, f16::from_f64, &stream);

                dev_op
                    .launch(
                        &args(
                            F16,
                            batch,
                            m,
                            n,
                            k,
                            c.as_mut_ptr().cast(),
                            a.as_ptr().cast(),
                            b.as_ptr().cast(),
                        ),
                        &mut [],
                        &stream,
                    )
                    .unwrap();

                let mut ans = vec![f16::ZERO; batch * m * n];
                dev.memcpy_d2h(&mut ans, &c);
                ans
            };

            let mut c_ref = c;
            cpu_op
                .launch(
                    &args(
                        F64,
                        batch,
                        m,
                        n,
                        k,
                        c_ref.as_mut_ptr().cast(),
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                    ),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = c_ref
                .into_par_iter()
                .zip(c_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 5e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
