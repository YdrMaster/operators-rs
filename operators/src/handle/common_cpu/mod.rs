pub struct Handle;
pub struct ThisThread;

impl common::Handle for Handle {
    type Byte = u8;
    type Queue<'ctx> = ThisThread;
}
