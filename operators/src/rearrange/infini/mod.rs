use super::{args::Scheme, Args, Rearrange};
use crate::{infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError};
use digit_layout::types;
use infini_op::{infiniop, AsRaw, Handle};
use std::{ptr::null_mut, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
}

impl Rearrange<Device> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    #[inline]
    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.handle.clone(),
        }
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
        let scheme = Scheme::new(args)?;
        let shape: Vec<_> = scheme.shape().map(|x| x as _).collect();
        let dst_strides: Vec<_> = scheme.dst_strides().iter().map(|&x| x as _).collect();
        let src_strides: Vec<_> = scheme.src_strides().iter().map(|&x| x as _).collect();

        let dst = infini_op::Tensor::new(types::U8, &shape, &dst_strides);
        let src = infini_op::Tensor::new(types::U8, &shape, &src_strides);

        let mut ptr = null_mut();
        infiniop!(infiniopCreateRearrangeDescriptor(
            self.handle.as_raw(),
            &mut ptr,
            dst.as_raw(),
            src.as_raw(),
        ));
        infiniop!(infiniopRearrange(
            ptr,
            args.dst_base.cast(),
            args.src_base.cast(),
            queue_alloc.queue().as_void_ptr()
        ));
        infiniop!(infiniopDestroyRearrangeDescriptor(ptr));
        Ok(())
    }
}
