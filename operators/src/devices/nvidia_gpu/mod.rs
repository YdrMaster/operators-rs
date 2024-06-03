pub struct Device;

impl crate::Device for Device {
    type Byte = cuda::DevByte;
    type Context = std::sync::Arc<cuda::Context>;
}
