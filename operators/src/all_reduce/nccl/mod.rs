use super::{AllReduce, Args, ReduceOp};
use crate::{
    nvidia_gpu::{Gpu, NcclNode},
    shape_mismatch,
    utils::{sizeof, type_distinct},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    nccl: Arc<nccl::Communicator>,
}

impl AllReduce<Gpu, NcclNode> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = NcclNode;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            nccl: node.nccl.clone(),
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
        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
            op,
        } = args;
        let dt = type_distinct(&[dst_layout.dt(), src_layout.dt()])?;
        let dst = dst_layout
            .shape()
            .iter()
            .map(|x| *x.get_static().unwrap())
            .product::<usize>();
        let src = src_layout
            .shape()
            .iter()
            .map(|x| *x.get_static().unwrap())
            .product::<usize>();
        if dst != src {
            return Err(shape_mismatch("").into());
        }
        let len = dst * sizeof(dt)?;
        self.nccl.all_reduce(
            unsafe { from_raw_parts_mut(*dst_base, len) },
            Some(unsafe { from_raw_parts(*src_base, len) }),
            dt,
            convert_enum(*op),
            queue_alloc.queue(),
        );
        Ok(())
    }
}

#[inline(always)]
fn convert_enum(op: ReduceOp) -> nccl::ReduceType {
    use nccl::ReduceType::*;
    match op {
        ReduceOp::Sum => ncclSum,
        ReduceOp::Prod => ncclProd,
        ReduceOp::Min => ncclMin,
        ReduceOp::Max => ncclMax,
        ReduceOp::Mean => ncclAvg,
    }
}
