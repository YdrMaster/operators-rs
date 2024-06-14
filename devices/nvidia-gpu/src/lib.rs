#![cfg(detected_cuda)]
#![deny(warnings)]

pub extern crate cublas;
pub extern crate cuda;

pub struct Device;

impl common::Device for Device {
    type Byte = cuda::DevByte;
    type Queue<'ctx> = cuda::Stream<'ctx>;
}

mod module;
pub use module::__global__;

mod cublas_handle;
pub use cublas_handle::{pools as preload_cublas, use_cublas};

fn contexts() -> &'static [cuda::Context] {
    use cuda::{Context, Device as Gpu};
    use std::sync::OnceLock;

    static CONTEXTS: OnceLock<Vec<Context>> = OnceLock::new();
    CONTEXTS.get_or_init(|| {
        cuda::init();
        (0..Gpu::count())
            .map(|i| Gpu::new(i as _).retain_primary())
            .collect()
    })
}
