use super::{args::Meta, Args, Broadcast};
use crate::{
    cuda::{Gpu, NcclNode},
    rearrange, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    nccl: Arc<nccl::Communicator>,
}

impl Broadcast<Gpu, NcclNode> for Operator {}

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
        let Meta { size } = args.meta()?;
        let &Args {
            pair: rearrange::Args {
                dst_base, src_base, ..
            },
            root,
            ..
        } = args;
        self.nccl.broadcast(
            unsafe { from_raw_parts_mut(dst_base, size) },
            Some(unsafe { from_raw_parts(src_base, size) }),
            root as _,
            queue_alloc.queue(),
        );
        Ok(())
    }
}
