use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use infini_rt::{DevBlob, DevByte, DeviceType, Stream};
use std::{ops::Deref, sync::Arc};

mod ccl;
pub use ccl::InfiniNode;

#[derive(Clone)]
pub struct Device {
    device: infini_rt::Device,
    handle: Arc<infini_op::Handle>,
}

impl Device {
    #[inline]
    pub fn cpu() -> Self {
        Self::new(infini_rt::DEVICE_CPU, 0)
    }

    #[inline]
    pub fn nv_gpu(id: usize) -> Self {
        Self::new(infini_rt::DEVICE_NVIDIA, id)
    }

    #[inline]
    pub fn cambricon_mlu(id: usize) -> Self {
        Self::new(infini_rt::DEVICE_CAMBRICON, id)
    }

    #[inline]
    pub fn ascend_npu(id: usize) -> Self {
        Self::new(infini_rt::DEVICE_ASCEND, id)
    }

    fn new(ty: infini_rt::DeviceType, id: usize) -> Self {
        use infini_op::bindings::Device as Ty;
        Self {
            device: infini_rt::Device { ty, id: id as _ },
            handle: Arc::new(infini_op::Handle::new(
                match ty {
                    infini_rt::DEVICE_CPU => Ty::DevCpu,
                    infini_rt::DEVICE_NVIDIA => Ty::DevNvGpu,
                    infini_rt::DEVICE_CAMBRICON => Ty::DevCambriconMlu,
                    infini_rt::DEVICE_ASCEND => Ty::DevAscendNpu,
                    _ => unreachable!("unknown device type"),
                },
                id as _,
            )),
        }
    }

    #[inline]
    pub(crate) fn device_type(&self) -> DeviceType {
        self.device.ty
    }

    #[inline]
    pub(crate) fn handle(&self) -> &Arc<infini_op::Handle> {
        &self.handle
    }
}

impl Deref for Device {
    type Target = infini_rt::Device;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
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

/// 并行转换类型并异步拷贝到显存。
#[cfg(test)]
pub(crate) fn cast_load<'ctx, T, U, F>(val: &[T], f: F, stream: &Stream) -> DevBlob
where
    T: Sync + Copy,
    U: Send + Copy,
    F: Sync + Fn(T) -> U,
{
    let mut host = stream.get_device().malloc_host::<U>(val.len());
    let host = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), val.len()) };
    host.into_iter().zip(val).for_each(|(y, x)| *y = f(*x));
    let ans = stream.from_host(host);
    stream.synchronize();
    ans
}
