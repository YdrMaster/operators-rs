use super::{args::Meta, AllReduce, Args, ReduceOp};
use crate::{
    infini::{Device, InfiniNode},
    rearrange, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::types as ty;
use infini_ccl::bindings::InfiniDataType_t;
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    comm: Arc<infini_ccl::Comm>,
}

impl AllReduce<Device, InfiniNode> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = InfiniNode;
    type Args = Args<Device>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            comm: node.comm.clone(),
        }
    }

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
        let Meta { dt, size } = args.meta()?;
        let &Args {
            pair: rearrange::Args {
                dst_base, src_base, ..
            },
            op,
            ..
        } = args;

        assert_eq!(op, ReduceOp::Sum);
        let len = dt.nbytes() * size;
        self.comm.allreduce_sum(
            unsafe { from_raw_parts_mut(dst_base, len) },
            unsafe { from_raw_parts(src_base, len) },
            match dt {
                ty::F16 => InfiniDataType_t::INFINI_F16,
                ty::F32 => InfiniDataType_t::INFINI_F32,
                _ => todo!(),
            },
            queue_alloc.queue(),
        );
        Ok(())
    }
}
