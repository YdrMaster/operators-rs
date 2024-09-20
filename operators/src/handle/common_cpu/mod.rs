pub use dev_mempool::{Blob, ThisThread};

pub struct Handle;

impl common::Handle for Handle {
    type Byte = u8;
    type DevMem<'ctx> = Blob;
    type Queue<'ctx> = ThisThread;
}
