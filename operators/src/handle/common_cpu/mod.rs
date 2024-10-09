use crate::{Hardware, QueueAlloc, QueueOf, TopoNode};
use dev_mempool::Blob;

#[derive(Clone, Copy, Debug)]
pub struct Cpu;

#[derive(Clone, Copy, Debug)]
pub struct Threads {
    id: usize,
    rank: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ThisThread;

impl Hardware for Cpu {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}

impl TopoNode<Cpu> for Threads {
    #[inline]
    fn processor(&self) -> &Cpu {
        &Cpu
    }
    #[inline]
    fn rank(&self) -> usize {
        self.id
    }
    #[inline]
    fn group_size(&self) -> usize {
        self.rank
    }
}

impl QueueAlloc for ThisThread {
    type Hardware = Cpu;
    type DevMem = Blob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
