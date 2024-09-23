use crate::{Hardware, QueueAlloc, QueueOf};
use dev_mempool::Blob;

#[derive(Clone, Copy, Debug)]
pub struct Cpu;

#[derive(Clone, Copy, Debug)]
pub struct ThisThread;

impl Hardware for Cpu {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}

impl QueueAlloc for ThisThread {
    type Hardware = Cpu;
    type DevMem = Blob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
