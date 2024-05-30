use crate::Device;

pub struct CommonCpu;

impl Device for CommonCpu {
    type Byte = u8;
    type Context = ();
}

// pub struct NvidiaGpu;

// impl Device for NvidiaGpu {
//     type Byte = u8;
//     type Context = ();
// }
