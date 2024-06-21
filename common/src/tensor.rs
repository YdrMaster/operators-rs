use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{copy_nonoverlapping, NonNull},
};

use digit_layout::DigitLayout;

/// | field   | type          |
/// |:-------:|:-------------:|
/// | dt      | DigitLayout    |
/// | udim    | u32           |
/// | offset  | usize         |
/// | shape   | [usize; ndim] |
/// | strides | [isize; ndim] |
#[repr(transparent)]
pub struct TensorLayout(NonNull<usize>);

impl TensorLayout {
    pub fn new(
        dt: DigitLayout,
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: usize,
    ) -> Self {
        let shape = shape.as_ref();
        let strides = strides.as_ref();
        assert_eq!(shape.len(), strides.len());

        let layout = Layout::array::<usize>(2 + shape.len() * 2).unwrap();
        unsafe {
            let ptr = alloc(layout);

            let cursor: *mut DigitLayout = ptr.cast();
            cursor.write(dt);
            let cursor: *mut u32 = cursor.add(1).cast();
            cursor.write(shape.len() as _);
            let cursor: *mut usize = cursor.add(1).cast();
            cursor.write(offset);
            let cursor: *mut usize = cursor.add(1).cast();
            copy_nonoverlapping(shape.as_ptr(), cursor, shape.len());
            let cursor: *mut isize = cursor.add(shape.len()).cast();
            copy_nonoverlapping(strides.as_ptr(), cursor, strides.len());

            Self(NonNull::new_unchecked(ptr as _))
        }
    }

    #[inline]
    pub fn dt(&self) -> DigitLayout {
        unsafe { *self.0.cast().as_ref() }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        unsafe { *self.0.cast::<u32>().as_ptr().add(1) as _ }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        unsafe { *self.0.cast::<usize>().as_ptr().add(1) }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        let len = self.ndim();
        unsafe {
            let ptr = self.0.cast::<usize>().as_ptr().add(2);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        let len = self.ndim();
        unsafe {
            let ptr = self.0.cast::<isize>().as_ptr().add(2 + len);
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for TensorLayout {
    #[inline]
    fn drop(&mut self) {
        let layout = Layout::array::<usize>(2 + self.ndim()).unwrap();
        unsafe { dealloc(self.0.cast().as_ptr(), layout) }
    }
}
