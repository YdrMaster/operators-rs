use digit_layout::layout;
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
};

#[repr(C)]
pub struct KVPair<T> {
    pub idx: u32,
    pub val: u32,
    pub _phantom: PhantomData<T>,
}

impl<T: Copy> KVPair<T> {
    layout!(LAYOUT u(32)x(2));

    pub fn new(idx: u32, val: T) -> Self {
        const { assert!(size_of::<T>() <= size_of::<u32>()) }
        const { assert!(align_of::<T>() <= align_of::<u32>()) }

        let mut val_bytes = 0;
        let ptr = std::ptr::from_mut(&mut val_bytes).cast::<T>();
        unsafe { ptr.write(val) };

        Self {
            idx,
            val: val_bytes,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn into_raw(self) -> KVPair<()> {
        KVPair {
            idx: self.idx,
            val: self.val,
            _phantom: PhantomData,
        }
    }
}
