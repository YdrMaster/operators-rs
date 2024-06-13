#![cfg(detected_cuda)]
#![deny(warnings)]

pub extern crate cuda;

pub struct Device;

impl common::Device for Device {
    type Byte = cuda::DevByte;
    type Queue<'ctx> = cuda::Stream<'ctx>;
}

mod module;
pub use module::__global__;

mod cublas;
