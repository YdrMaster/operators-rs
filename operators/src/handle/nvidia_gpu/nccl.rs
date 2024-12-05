use super::{Config, Gpu};
use crate::TopoNode;
use nccl::Communicator;
use std::sync::Arc;

pub struct NcclNode {
    gpu: Gpu,
    pub(crate) nccl: Arc<Communicator>,
}

impl NcclNode {
    pub fn new(comm: Communicator, config: Config) -> Self {
        Self {
            gpu: Gpu::new(comm.device().retain_primary(), config),
            nccl: Arc::new(comm),
        }
    }
}

impl TopoNode<Gpu> for NcclNode {
    #[inline]
    fn processor(&self) -> &Gpu {
        &self.gpu
    }
    #[inline]
    fn rank(&self) -> usize {
        self.nccl.rank()
    }
    #[inline]
    fn group_size(&self) -> usize {
        self.nccl.count()
    }
}
