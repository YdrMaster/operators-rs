use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use clrt::{CommandQueue, Context, SvmBlob, SvmByte};

#[repr(transparent)]
pub struct ClDevice(Context);

impl Hardware for ClDevice {
    type Byte = SvmByte;
    type Queue<'ctx> = CommandQueue;
}

impl ClDevice {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(context)
    }

    #[inline]
    pub(crate) fn context(&self) -> &Context {
        &self.0
    }
}

impl Alloc<SvmBlob> for Context {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.malloc::<usize>(size)
    }

    #[inline]
    fn free(&self, _mem: SvmBlob) {}
}

impl Alloc<SvmBlob> for CommandQueue {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.ctx().malloc::<usize>(size)
    }

    #[inline]
    fn free(&self, mem: SvmBlob) {
        self.free(mem, None)
    }
}

impl QueueAlloc for CommandQueue {
    type Hardware = ClDevice;
    type DevMem = SvmBlob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
