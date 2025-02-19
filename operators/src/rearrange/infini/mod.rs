use super::{args::Scheme, Args, Rearrange};
use crate::{infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError};
use digit_layout::types;
use infini_op::{infiniop, AsRaw, Descriptor, Handle};
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator(Arc<Handle>);

impl Rearrange<Device> for Operator {}

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
        use std::iter::once;

        let scheme = Scheme::new(args)?;
        if scheme.ndim() == 0 {
            let unit = scheme.unit();
            let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let src = unsafe { from_raw_parts(args.src_base, unit) };
            queue_alloc.queue().memcpy_d2d(dst, src);
            return Ok(());
        }

        let scheme = scheme.distribute_unit((0..=5).rev().map(|n| 32 * (1 << n)));
        let unit = scheme.unit();

        let dst = infini_op::Tensor::new(
            types::U8,
            scheme.shape().chain(once(unit)),
            scheme.dst_strides().iter().cloned().chain(once(1)),
        );
        let src = infini_op::Tensor::new(
            types::U8,
            scheme.shape().chain(once(unit)),
            scheme.src_strides().iter().cloned().chain(once(1)),
        );

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateRearrangeDescriptor(
                    self.0.as_raw(),
                    ptr,
                    dst.as_raw(),
                    src.as_raw(),
                ))
            },
            infini_op::bindings::infiniopDestroyRearrangeDescriptor,
        );
        infiniop!(infiniopRearrange(
            descriptor.as_raw(),
            args.dst_base.cast(),
            args.src_base.cast(),
            queue_alloc.queue().as_void_ptr(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Device, Operator};
    use crate::{ConstPtr, Hardware, MutPtr, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            dst_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        shape: &[usize],
        s_src: &[isize],
        s_dst: &[isize],
        src_base: ConstPtr<H>,
        dst_base: MutPtr<H>,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new(dt, shape, s_dst),
            dst_base,
            src_layout: TensorLayout::new(dt, shape, s_src),
            src_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::common_cpu::{Cpu, ThisThread};
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;

        let dt = ty::U32;
        let nh = 32;
        let seq = 7;
        let dh = 128;

        infini_rt::init(infini_rt::DEVICE_CPU);
        let dev = Device::cpu();

        let mut cpu_op = RefOp::new(&Cpu);
        let mut dev_op = Operator::new(&dev);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        dev_op.scheme(&dyn_args(dt), 0).unwrap();

        let mut src = vec![0u32; nh * seq * dh];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, ele);
        let s_dst =
            ArrayLayout::<3>::new_contiguous(&[seq, nh, dh], BigEndian, ele).transpose(&[1, 0]);

        let dst_ans = {
            let stream = dev.stream();
            let src = stream.from_host(&src);
            let mut dst = stream.malloc::<u8>(src.len());
            dev_op
                .launch(
                    &args(
                        dt,
                        &[nh, seq, dh],
                        s_src.strides(),
                        s_dst.strides(),
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![0u32; nh * seq * dh];
            dev.memcpy_d2h(&mut host, &dst);
            host
        };

        let mut dst_ref = vec![0u32; seq * nh * dh];
        cpu_op
            .launch(
                &args(
                    dt,
                    &[nh, seq, dh],
                    s_src.strides(),
                    s_dst.strides(),
                    src.as_ptr().cast(),
                    dst_ref.as_mut_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        assert_eq!(dst_ans, dst_ref);
    }
}
