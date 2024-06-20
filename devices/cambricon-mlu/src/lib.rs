#![cfg(detected_neuware)]
#![deny(warnings)]

pub extern crate cndrv;

pub struct Device;

impl common::Device for Device {
    type Byte = cndrv::DevByte;
    type Queue<'ctx> = cndrv::Queue<'ctx>;
}
