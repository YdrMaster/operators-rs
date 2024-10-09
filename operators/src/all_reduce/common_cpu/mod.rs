use super::{AllReduce, Args};
use crate::{
    common_cpu::{Cpu, Threads},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};

pub struct Operator;

impl AllReduce<Cpu, Threads> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Threads;
    type Args = Args<Cpu>;

    fn new(_node: &Self::TopoNode) -> Self {
        Self
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
