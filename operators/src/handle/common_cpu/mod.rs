mod inproc_node;

use crate::{Alloc, Blob, Hardware, QueueAlloc, QueueOf};

pub use inproc_node::InprocNode;

#[derive(Clone, Copy, Debug)]
pub struct Cpu;

#[derive(Clone, Copy, Debug)]
pub struct ThisThread;

impl Hardware for Cpu {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}

impl<T> Alloc<Blob> for T {
    #[inline]
    fn alloc(&self, size: usize) -> Blob {
        Blob::new(size)
    }

    #[inline]
    fn free(&self, _mem: Blob) {}
}

impl QueueAlloc for ThisThread {
    type Hardware = Cpu;
    type DevMem = Blob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
