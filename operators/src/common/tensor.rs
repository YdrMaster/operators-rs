use crate::MaybeDyn;
use digit_layout::DigitLayout;
use ndarray_layout::ArrayLayout;
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
};

/// | field    | type          |
/// |:--------:|:-------------:|
/// | dt       | DigitLayout   |
/// | ndim     | u64           |
/// | shape    | [usize; ndim] |
/// | strides  | [isize; ndim] |
#[repr(transparent)]
pub struct TensorLayout(NonNull<usize>);

impl TensorLayout {
    pub fn new_dyn(
        dt: DigitLayout,
        shape: &[MaybeDyn<usize>],
        strides: &[MaybeDyn<isize>],
    ) -> Self {
        let shape: &[usize] = unsafe { std::mem::transmute(shape) };
        let strides: &[isize] = unsafe { std::mem::transmute(strides) };
        Self::new(dt, shape, strides)
    }

    pub fn new(dt: DigitLayout, shape: &[usize], strides: &[isize]) -> Self {
        assert_eq!(shape.len(), strides.len());

        unsafe {
            let ptr = alloc(Self::layout(shape.len()));

            let cursor: *mut DigitLayout = ptr.cast();
            cursor.write(dt);
            let cursor: *mut u64 = cursor.add(1).cast();
            cursor.write(shape.len() as _);
            let cursor: *mut usize = cursor.add(1).cast();
            copy_nonoverlapping(shape.as_ptr(), cursor, shape.len());
            let cursor: *mut isize = cursor.add(shape.len()).cast();
            copy_nonoverlapping(strides.as_ptr(), cursor, strides.len());

            Self(NonNull::new_unchecked(ptr as _))
        }
    }

    pub fn new_contiguous(dt: DigitLayout, shape: &[usize]) -> Self {
        let mut strides = shape
            .iter()
            .rev()
            .scan(dt.nbytes() as isize, |mul, &d| {
                let stride = *mul;
                *mul *= d as isize;
                Some(stride)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        Self::new(dt, shape, &strides)
    }

    #[inline]
    pub fn from_arr<const N: usize>(dt: DigitLayout, arr: &ArrayLayout<N>) -> Self {
        Self::new(dt, arr.shape(), arr.strides())
    }

    #[inline]
    pub fn dt(&self) -> DigitLayout {
        let ptr = self.0.cast();
        unsafe { *ptr.as_ref() }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        let ptr = self.0.cast::<u64>().as_ptr();
        unsafe { *ptr.add(1) as _ }
    }

    #[inline]
    pub fn shape(&self) -> &[MaybeDyn<usize>] {
        let ptr = self.0.cast::<MaybeDyn<usize>>().as_ptr();
        let len = self.ndim();
        unsafe { from_raw_parts(ptr.add(2), len) }
    }

    #[inline]
    pub fn strides(&self) -> &[MaybeDyn<isize>] {
        let ptr = self.0.cast::<MaybeDyn<isize>>().as_ptr();
        let len = self.ndim();
        unsafe { from_raw_parts(ptr.add(2 + len), len) }
    }

    #[inline(always)]
    fn layout(ndim: usize) -> Layout {
        Layout::array::<usize>(2 + ndim * 2).unwrap()
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
