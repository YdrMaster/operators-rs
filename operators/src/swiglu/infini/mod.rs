use super::{args::Meta, Args, Swiglu};
use crate::{get_static, infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError};
use infini_op::{infiniop, AsRaw, Descriptor, Handle};
use std::sync::Arc;

pub struct Operator(Arc<Handle>);

impl Swiglu<Device> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    #[inline]
    fn new(node: &Self::TopoNode) -> Self {
        Self(node.handle().clone())
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
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt, n, d } = args.meta()?;
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[gns, gds] = gate_layout.strides() else {
            unreachable!()
        };
        let &[uns, uds] = up_layout.strides() else {
            unreachable!()
        };

        get_static! {
             n   d
            gns gds
            uns uds
        }

        let gate = infini_op::Tensor::new(dt, [n, d], [gns, gds]);
        let up = infini_op::Tensor::new(dt, [n, d], [uns, uds]);

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateSwiGLUDescriptor(
                    self.0.as_raw(),
                    ptr,
                    gate.as_raw(),
                    up.as_raw(),
                    gate.as_raw(),
                ))
            },
            infini_op::bindings::infiniopDestroySwiGLUDescriptor,
        );
        infiniop!(infiniopSwiGLU(
            descriptor.as_raw(),
            gate_base.cast(),
            up_base.cast(),
            gate_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Device, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::{null, null_mut};
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,
        gate_base: *mut H::Byte,
        up_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt, &[n, d]);
        Args {
            gate_layout: layout.clone(),
            gate_base,
            up_layout: layout,
            up_base,
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

        let n = 5632;
        let d = 2048;

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let mut cpu_op = RefOp::new(&Cpu);
        let mut dev_op = Operator::new(&dev);
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        dev_op.scheme(&dyn_args(F16), 0).unwrap();

        let mut gate = vec![0.0f64; n * d];
        let mut up = vec![0.0f64; n * d];
        rand::rng().fill(&mut gate[..]);
        rand::rng().fill(&mut up[..]);
        let up = up;

        let gate_ans = {
            let stream = dev.stream();
            let mut gate = cast_load(&gate, f16::from_f64, &stream);
            let up = cast_load(&up, f16::from_f64, &stream);
            dev_op
                .launch(
                    &args(F16, n, d, gate.as_mut_ptr().cast(), up.as_ptr().cast()),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; n * d];
            dev.memcpy_d2h(&mut host, &gate);
            host
        };

        let mut gate_ref = gate;
        cpu_op
            .launch(
                &args(F64, n, d, gate_ref.as_mut_ptr().cast(), up.as_ptr().cast()),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = gate_ref
            .into_iter()
            .zip(gate_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
