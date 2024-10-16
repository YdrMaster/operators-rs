mod inproc_node;

pub use dev_mempool::Blob;
pub use inproc_node::InprocNode;

use crate::{Hardware, QueueAlloc, QueueOf};

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
