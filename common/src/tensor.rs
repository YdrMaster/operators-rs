use crate::Argument;
use digit_layout::DigitLayout;
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
};

/// | field    | type          |
/// |:--------:|:-------------:|
/// | dt       | DigitLayout   |
/// | ndim     | u32           |
/// | shape    | [usize; ndim] |
/// | strides  | [isize; ndim] |
#[repr(transparent)]
pub struct TensorLayout(NonNull<usize>);

impl TensorLayout {
    pub fn new(
        dt: DigitLayout,
        shape: impl AsRef<[Argument<usize>]>,
        strides: impl AsRef<[Argument<isize>]>,
    ) -> Self {
        let shape = shape.as_ref();
        let strides = strides.as_ref();
        assert_eq!(shape.len(), strides.len());

        unsafe {
            let ptr = alloc(Self::layout(shape.len()));

            let cursor: *mut DigitLayout = ptr.cast();
            cursor.write(dt);
            let cursor: *mut u32 = cursor.add(1).cast();
            cursor.write(shape.len() as _);
            let cursor: *mut Argument<usize> = cursor.add(1).cast();
            copy_nonoverlapping(shape.as_ptr(), cursor, shape.len());
            let cursor: *mut Argument<isize> = cursor.add(shape.len()).cast();
            copy_nonoverlapping(strides.as_ptr(), cursor, strides.len());

            Self(NonNull::new_unchecked(ptr as _))
        }
    }

    #[inline]
    pub fn dt(&self) -> DigitLayout {
        let ptr = self.0.cast();
        unsafe { *ptr.as_ref() }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        let ptr = self.0.cast::<u32>().as_ptr();
        unsafe { *ptr.add(1) as _ }
    }

    #[inline]
    pub fn shape(&self) -> &[Argument<usize>] {
        let ptr = self.0.cast::<Argument<usize>>().as_ptr();
        let len = self.ndim();
        unsafe { from_raw_parts(ptr.add(1), len) }
    }

    #[inline]
    pub fn strides(&self) -> &[Argument<isize>] {
        let ptr = self.0.cast::<Argument<isize>>().as_ptr();
        let len = self.ndim();
        unsafe { from_raw_parts(ptr.add(1 + len), len) }
    }

    #[inline(always)]
    fn layout(ndim: usize) -> Layout {
        Layout::array::<usize>(1 + ndim * 2).unwrap()
    }
}

impl Clone for TensorLayout {
    #[inline]
    fn clone(&self) -> Self {
        let layout = Self::layout(self.ndim());
        let src = self.0.cast::<u8>().as_ptr();
        unsafe {
            let dst = alloc(layout);
            copy_nonoverlapping(src, dst, layout.size());
            Self(NonNull::new_unchecked(dst as _))
        }
    }
}

impl Drop for TensorLayout {
    #[inline]
    fn drop(&mut self) {
        let ptr = self.0.cast().as_ptr();
        let layout = Self::layout(self.ndim());
        unsafe { dealloc(ptr, layout) }
    }
}
