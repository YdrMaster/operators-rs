use digit_layout::layout;
use std::{
    cmp::Ordering::{self, Equal},
    marker::PhantomData,
    mem::{align_of, size_of},
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KVPair<T = ()> {
    idx: u32,
    val: u32,
    _phantom: PhantomData<T>,
}

impl<T: Copy> KVPair<T> {
    layout!(LAYOUT u(32); 2);

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

    #[inline]
    pub const fn idx(&self) -> usize {
        self.idx as _
    }

    #[inline]
    pub const fn val(&self) -> T {
        let bytes = self.val.to_ne_bytes();
        unsafe { bytes.as_ptr().cast::<T>().read() }
    }
}

impl KVPair<f32> {
    #[inline]
    pub fn set_val(&mut self, val: f32) {
        self.val = val.to_bits();
    }
}

impl<T: PartialOrd + Copy> PartialEq for KVPair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Equal
    }
}
impl<T: PartialOrd + Copy> Eq for KVPair<T> {}
impl<T: PartialOrd + Copy> PartialOrd for KVPair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: PartialOrd + Copy> Ord for KVPair<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.val().partial_cmp(&other.val()) {
            Some(Equal) => match self.idx.cmp(&other.idx) {
                Equal => Equal,
                ord => ord,
            },
            Some(ord) => ord.reverse(),
            None => Equal,
        }
    }
}
