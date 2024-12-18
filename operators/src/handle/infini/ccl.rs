use super::Device;
use crate::TopoNode;
use infini_ccl::{bindings::DeviceType, Comm};
use std::{os::raw::c_uint, sync::Arc};

pub struct InfiniNode {
    rank: usize,
    group_size: usize,
    pub(crate) device: Device,
    pub(crate) comm: Arc<Comm>,
}

impl InfiniNode {
    pub fn cpu(n: usize) -> Vec<Self> {
        let device = Device::cpu();
        let indices = (0..n as c_uint).collect::<Vec<_>>();
        Comm::init_all(DeviceType::DEVICE_CPU, &indices)
            .into_iter()
            .enumerate()
            .map(|(id, comm)| Self {
                rank: id,
                group_size: n,
                device: device.clone(),
                comm: Arc::new(comm),
            })
            .collect()
    }

    pub fn nv_gpu(indices: &[c_uint]) -> Vec<Self> {
        Comm::init_all(DeviceType::DEVICE_NVIDIA, indices)
            .into_iter()
            .enumerate()
            .map(|(id, comm)| Self {
                rank: id,
                group_size: indices.len(),
                device: Device::nv_gpu(id),
                comm: Arc::new(comm),
            })
            .collect()
    }

    pub fn cambricon_mlu(indices: &[c_uint]) -> Vec<Self> {
        Comm::init_all(DeviceType::DEVICE_CAMBRICON, indices)
            .into_iter()
            .enumerate()
            .map(|(id, comm)| Self {
                rank: id,
                group_size: indices.len(),
                device: Device::cambricon_mlu(id),
                comm: Arc::new(comm),
            })
            .collect()
    }

    pub fn ascend_npu(indices: &[c_uint]) -> Vec<Self> {
        Comm::init_all(DeviceType::DEVICE_ASCEND, indices)
            .into_iter()
            .enumerate()
            .map(|(id, comm)| Self {
                rank: id,
                group_size: indices.len(),
                device: Device::ascend_npu(id),
                comm: Arc::new(comm),
            })
            .collect()
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
