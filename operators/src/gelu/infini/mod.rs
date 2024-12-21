use super::{args::Meta, Args, Gelu};
use crate::{get_static, infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError};
use infini_op::{infiniop, AsRaw, Descriptor, Handle};
use std::sync::Arc;

pub struct Operator(Arc<Handle>);

impl Gelu<Device> for Operator {}

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
