pub struct Device;
pub struct ThisThread;

impl crate::Device for Device {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}
