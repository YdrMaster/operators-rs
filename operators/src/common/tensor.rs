use digit_layout::DigitLayout;
use ndarray_layout::ArrayLayout;
use std::borrow::Cow;

#[derive(Clone)]
pub struct TensorLayout {
    pub dt: DigitLayout,
    pub layout: ArrayLayout<4>,
}

impl TensorLayout {
    pub fn new(dt: DigitLayout, shape: &[usize], strides: &[isize]) -> Self {
        Self {
            dt,
            layout: ArrayLayout::new(shape, strides, 0),
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
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    #[inline]
    pub fn shape_group(&self) -> &[usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn shape(&self) -> Cow<[usize]> {
        if self.dt.group_size() == 1 {
            Cow::Borrowed(self.layout.shape())
        } else {
            Cow::Owned(vec![])
        }
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }
}
