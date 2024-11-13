use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use ascendcl::{Context, CurrentCtx, DevByte, DevMem, Stream};

#[repr(transparent)]
pub struct Npu(Context);

impl Hardware for Npu {
    type Byte = DevByte;
    type Queue<'ctx> = Stream<'ctx>;
}

impl Npu {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(context)
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        self.0.apply(f)
    }
}

impl<'ctx> Alloc<DevMem<'ctx>> for &'ctx CurrentCtx {
    #[inline]
    fn alloc(&self, size: usize) -> DevMem<'ctx> {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevMem<'ctx>) {}
}

impl<'ctx> Alloc<DevMem<'ctx>> for Stream<'ctx> {
    #[inline]
    fn alloc(&self, size: usize) -> DevMem<'ctx> {
        self.ctx().malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevMem<'ctx>) {}
}

impl<'ctx> QueueAlloc for Stream<'ctx> {
    type Hardware = Npu;
    type DevMem = DevMem<'ctx>;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
