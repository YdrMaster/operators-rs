use std::{
    alloc::{alloc, dealloc, Layout},
    ops::{Deref, DerefMut},
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Blob {
    ptr: NonNull<u8>,
    len: usize,
}

impl Blob {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            ptr: NonNull::new(unsafe { alloc(layout(size)) }).unwrap(),
            len: size,
        }
    }
}

impl Drop for Blob {
    #[inline]
    fn drop(&mut self) {
        let &mut Blob { ptr, len } = self;
        unsafe { dealloc(ptr.as_ptr(), layout(len)) }
    }
}

#[inline(always)]
const fn layout(size: usize) -> Layout {
    unsafe { Layout::from_size_align_unchecked(size, align_of::<usize>()) }
}

impl Deref for Blob {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Blob {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
