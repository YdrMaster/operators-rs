pub trait DynVal {
    fn default_dyn() -> Self;
    fn is_dynamic(&self) -> bool;
}

impl DynVal for isize {
    #[inline]
    fn default_dyn() -> Self {
        Self::MAX
    }
    #[inline]
    fn is_dynamic(&self) -> bool {
        *self == Self::MAX
    }
}

impl DynVal for usize {
    #[inline]
    fn default_dyn() -> Self {
        Self::MAX
    }
    #[inline]
    fn is_dynamic(&self) -> bool {
        *self == Self::MAX
    }
}

impl DynVal for f32 {
    #[inline]
    fn default_dyn() -> Self {
        Self::INFINITY
    }
    #[inline]
    fn is_dynamic(&self) -> bool {
        self.is_infinite() && self.is_sign_positive()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct MaybeDyn<T>(pub T);

impl<T> From<T> for MaybeDyn<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: DynVal> MaybeDyn<T> {
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
pub fn dyn_<T: DynVal>() -> MaybeDyn<T> {
    MaybeDyn::dynamic()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MergeError {
    EmptyIter,
    NotMatch,
}

impl<T: DynVal + PartialEq> MaybeDyn<T> {
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

    pub fn get_all(slice: &[Self]) -> Option<&[T]> {
        if slice.iter().any(|arg| arg.is_dynamic()) {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) })
        }
    }
}

#[inline]
pub(crate) fn static_from<T: DynVal>(arg: &MaybeDyn<T>) -> Result<&T, SchemeError> {
    arg.get_static().ok_or_else(|| dyn_not_support(""))
}

macro_rules! get_static {
    ($($name:ident)*) => {
        $( let $name = *$crate::static_from(&$name)?; )*
    };
}

pub(crate) use get_static;

use super::{dyn_not_support, SchemeError};
