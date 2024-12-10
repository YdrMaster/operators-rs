use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use infini_rt::{DevBlob, DevByte, Stream};
use std::sync::Arc;

pub struct Device {
    pub(crate) device: infini_rt::Device,
    pub(crate) handle: Arc<infini_op::Handle>,
}

impl Hardware for Device {
    type Byte = DevByte;
    type Queue<'ctx> = Stream;
}

impl Alloc<DevBlob> for Device {
    #[inline]
    fn alloc(&self, size: usize) -> DevBlob {
        self.device.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevBlob) {}
}

impl Alloc<DevBlob> for Stream {
    #[inline]
    fn alloc(&self, size: usize) -> DevBlob {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, mem: DevBlob) {
        self.free(mem)
    }
}

impl QueueAlloc for Stream {
    type Hardware = Device;
    type DevMem = DevBlob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
