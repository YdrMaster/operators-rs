use super::{Args, Swiglu};
use crate::{ascend::Npu, ByteOf, LaunchError, QueueAlloc, SchemeError};

pub struct Operator;

impl Swiglu<Npu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Npu;
    type TopoNode = Npu;
    type Args = Args<Npu>;

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
