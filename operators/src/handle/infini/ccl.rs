use super::Device;
use crate::TopoNode;
use infini_ccl::{bindings::DeviceType, Comm};
use std::{os::raw::c_uint, sync::Arc};

pub struct InfiniNode {
    rank: usize,
    group_size: usize,
    pub(crate) device: Device,
    pub(crate) comm: Option<Arc<Comm>>,
}

impl InfiniNode {
    pub fn cpu(n: usize) -> Vec<Self> {
        let indices = (0..n as _).collect::<Vec<_>>();
        Self::new(&indices, DeviceType::DEVICE_CPU)
    }

    pub fn nv_gpu(indices: &[c_uint]) -> Vec<Self> {
        Self::new(indices, DeviceType::DEVICE_NVIDIA)
    }

    pub fn cambricon_mlu(indices: &[c_uint]) -> Vec<Self> {
        Self::new(indices, DeviceType::DEVICE_CAMBRICON)
    }

    pub fn ascend_npu(indices: &[c_uint]) -> Vec<Self> {
        Self::new(indices, DeviceType::DEVICE_ASCEND)
    }

    fn new(indices: &[c_uint], ty: DeviceType) -> Vec<Self> {
        let confused: infini_rt::DeviceType = unsafe { std::mem::transmute(ty) };
        if let &[id] = indices {
            vec![Self {
                rank: 0,
                group_size: 1,
                device: Device::new(confused, id as _),
                comm: None,
            }]
        } else {
            Comm::init_all(ty, indices)
                .into_iter()
                .zip(indices)
                .enumerate()
                .map(|(idx, (comm, &id))| Self {
                    rank: idx,
                    group_size: indices.len(),
                    device: Device::new(confused, id as _),
                    comm: Some(Arc::new(comm)),
                })
                .collect()
        }
    }
}

impl TopoNode<Device> for InfiniNode {
    #[inline]
    fn processor(&self) -> &Device {
        &self.device
    }
    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }
    #[inline]
    fn group_size(&self) -> usize {
        self.group_size
    }
}
