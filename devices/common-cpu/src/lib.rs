#![deny(warnings)]

pub struct Device;
pub struct ThisThread;

impl common::Device for Device {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}
