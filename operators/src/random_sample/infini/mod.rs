use super::{Args, Indices, RandomSample};
use crate::{infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError};

pub struct Operator;

impl RandomSample<Device> for Operator {
    fn build_indices<QA>(_n: usize, _queue_alloc: &QA) -> Indices<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        todo!()
    }
}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    fn new(_node: &Self::TopoNode) -> Self {
        todo!()
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        todo!()
    }

    fn launch<QA>(
        &self,
        _args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        todo!()
    }
}
