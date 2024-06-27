use std::ops::{Deref, DerefMut};

pub trait ArgVal {
    fn default_dyn() -> Self;
    fn is_dynamic(&self) -> bool;
}

impl ArgVal for isize {
    #[inline]
    fn default_dyn() -> Self {
        Self::MAX
    }
    #[inline]
    fn is_dynamic(&self) -> bool {
        *self == Self::MAX
    }
}

impl ArgVal for usize {
    #[inline]
    fn default_dyn() -> Self {
        Self::MAX
    }
    #[inline]
    fn is_dynamic(&self) -> bool {
        *self == Self::MAX
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Argument<T>(T);

impl<T> From<T> for Argument<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for Argument<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Argument<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Argument<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: ArgVal> Argument<T> {
    #[inline]
    pub fn dynamic() -> Self {
        Self(T::default_dyn())
    }
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.0.is_dynamic()
    }
    #[inline]
    pub fn get_static(&self) -> Option<&T> {
        if !self.is_dynamic() {
            Some(&self.0)
        } else {
            None
        }
    }
}

#[inline(always)]
pub fn dyn_<T: ArgVal>() -> Argument<T> {
    Argument::dynamic()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MergeError {
    EmptyIter,
    NotMatch,
}

impl<T: ArgVal + PartialEq> Argument<T> {
    pub fn merge<'a>(iter: impl IntoIterator<Item = &'a Self>) -> Result<&'a Self, MergeError> {
        let mut iter = iter.into_iter();
        let mut acc = iter.next().ok_or(MergeError::EmptyIter)?;
        for it in iter {
            if it.is_dynamic() {
                // Nothing to do
            } else if acc.is_dynamic() {
                acc = it;
            } else if acc.0 != it.0 {
                return Err(MergeError::NotMatch);
            }
        }
        Ok(acc)
    }
}
