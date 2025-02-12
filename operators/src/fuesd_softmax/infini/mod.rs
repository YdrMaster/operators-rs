use infini_op::{infiniop, AsRaw, Descriptor};

use super::{args::Meta, Args, FusedSoftmax};
use crate::{
    fuesd_softmax::args::AttnMask, get_static, infini::Device, ByteOf, LaunchError, QueueAlloc,
    SchemeError, Workspace,
};

pub struct Operator(Device);

impl FusedSoftmax<Device> for Operator {}

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
        let Meta { dt } = args.meta()?;
        let Args {
            att_mask,
            att_layout,
            att_base,
        } = args;
        if !matches!(att_mask, AttnMask::Causal) {
            todo!()
        }
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        let &[sh, ss, sa] = att_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nh seq_len att_len
            sh ss      sa
        }

        let att = infini_op::Tensor::new(dt, [nh, seq_len, att_len], [sh, ss, sa]);
        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateCausalSoftmaxDescriptor(
                    self.0.handle().as_raw(),
                    ptr,
                    att.as_raw(),
                ))
            },
            infini_op::bindings::infiniopDestroyCausalSoftmaxDescriptor,
        );
        let mut workspace_size = 0;
        infiniop!(infiniopGetCausalSoftmaxWorkspaceSize(
            descriptor.as_raw(),
            &mut workspace_size
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);
        infiniop!(infiniopCausalSoftmax(
            descriptor.as_raw(),
            workspace.as_mut_ptr().cast(),
            workspace_size,
            att_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, AttnMask, Device, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::null_mut;
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]),
            att_base: null_mut(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        seq_len: usize,
        att_len: usize,
        att_base: *mut H::Byte,
    ) -> Args<H> {
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_contiguous(dt, &[nh, seq_len, att_len]),
            att_base,
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

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let mut cpu_op = RefOp::new(&Cpu);
        let mut dev_op = Operator::new(&dev);
        cpu_op.scheme(&dyn_args(ty::F64), 0).unwrap();
        dev_op.scheme(&dyn_args(ty::F16), 0).unwrap();

        let nh = 32;
        for (seq_len, att_len) in [(1, 511), (1, 2048), (7, 511), (7, 2048)] {
            let mut att = vec![0.0f64; nh * seq_len * att_len];
            rand::rng().fill(&mut att[..]);

            let att_ans = {
                let stream = dev.stream();
                let mut att = cast_load(&att, f16::from_f64, &stream);
                dev_op
                    .launch(
                        &args(ty::F16, nh, seq_len, att_len, att.as_mut_ptr().cast()),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; nh * seq_len * att_len];
                dev.memcpy_d2h(&mut host, &att);
                host
            };

            let mut att_ref = att;
            cpu_op
                .launch(
                    &args(ty::F64, nh, seq_len, att_len, att_ref.as_mut_ptr().cast()),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = att_ref
                .into_iter()
                .zip(att_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
