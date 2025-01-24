use super::Gpu;
use crate::{Alloc, OffsetCalculator, QueueAlloc, QueueOf};
use cuda::{CurrentCtx, DevByte, DevMem, Stream};
use std::{
    cell::RefCell,
    ops::{Deref, DerefMut, Range},
    rc::Rc,
};

pub struct StreamMemPool<'ctx> {
    stream: Stream<'ctx>,
    mem_pool: Rc<RefCell<MemPool<'ctx>>>,
}

pub struct MemPoolBlob<'ctx> {
    mem_pool: Rc<RefCell<MemPool<'ctx>>>,
    range: Range<usize>,
}

struct MemPool<'ctx> {
    pool: Vec<DevMem<'ctx>>,
    recorder: OffsetCalculator,
}

impl Deref for MemPoolBlob<'_> {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.range.start as _, self.range.len()) }
    }
}

impl DerefMut for MemPoolBlob<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.range.start as _, self.range.len()) }
    }
}

impl<'ctx> StreamMemPool<'ctx> {
    pub fn new(stream: Stream<'ctx>) -> Self {
        let alignment = stream.ctx().dev().alignment();
        Self {
            stream,
            mem_pool: Rc::new(RefCell::new(MemPool {
                pool: Vec::new(),
                recorder: OffsetCalculator::new(if alignment == 0 { 256 } else { alignment }),
            })),
        }
    }

    pub fn put(&self, size: usize) {
        let blob = self.stream.ctx().malloc::<u8>(size);
        let area = blob.as_ptr_range();
        let mut mem_pool = self.mem_pool.borrow_mut();
        mem_pool.pool.push(blob);
        mem_pool.recorder.put(&(area.start as _..area.end as _));
    }
}

impl<'ctx> Alloc<MemPoolBlob<'ctx>> for StreamMemPool<'ctx> {
    #[inline]
    fn alloc(&self, size: usize) -> MemPoolBlob<'ctx> {
        let range = self
            .mem_pool
            .borrow_mut()
            .recorder
            .take(size)
            .expect("out of memory");
        MemPoolBlob {
            mem_pool: self.mem_pool.clone(),
            range,
        }
    }

    #[inline]
    fn free(&self, mem: MemPoolBlob<'ctx>) {
        assert!(Rc::ptr_eq(&self.mem_pool, &mem.mem_pool));
        self.mem_pool.borrow_mut().recorder.put(&mem.range)
    }
}

impl<'ctx> QueueAlloc for StreamMemPool<'ctx> {
    type Hardware = Gpu;
    type DevMem = MemPoolBlob<'ctx>;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        &self.stream
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

#[cfg(use_nvidia)]
impl<'ctx> Alloc<DevMem<'ctx>> for Stream<'ctx> {
    #[inline]
    fn alloc(&self, size: usize) -> DevMem<'ctx> {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, mem: DevMem<'ctx>) {
        mem.drop_on(self)
    }
}

#[cfg(use_nvidia)]
impl<'ctx> QueueAlloc for Stream<'ctx> {
    type Hardware = Gpu;
    type DevMem = DevMem<'ctx>;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
