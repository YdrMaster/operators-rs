use crate::Hardware;
use clrt::Context;

#[repr(transparent)]
pub struct ClDevice(Context);

impl Hardware for ClDevice {
    type Byte = clrt::SvmByte;
    type Queue<'ctx> = clrt::CommandQueue;
}

impl ClDevice {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(context)
    }
}
