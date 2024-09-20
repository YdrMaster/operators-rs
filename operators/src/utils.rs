#![allow(unused)]

use common::{
    dyn_not_support, shape_mismatch, type_mismatch, type_not_support, ArgVal, Argument, Handle,
    ParamError, SchemeErrorKind,
};
use digit_layout::DigitLayout;

pub(crate) type MutPtr<H> = *mut <H as Handle>::Byte;
pub(crate) type ConstPtr<H> = *const <H as Handle>::Byte;

#[inline]
pub(crate) const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

macro_rules! get_static {
    ($($name:ident)*) => {
        $(
            let $name = *crate::utils::static_from(&$name)?;
        )*
    };
}

#[inline]
pub(crate) fn static_from<T: ArgVal>(arg: &Argument<T>) -> Result<&T, ParamError> {
    arg.get_static().ok_or_else(|| dyn_not_support(""))
}

#[inline]
pub(crate) fn sizeof(dt: DigitLayout) -> Result<usize, ParamError> {
    dt.nbytes()
        .ok_or_else(|| type_not_support(format!("{dt} not supported")))
}

#[inline]
pub(crate) fn rank_not_support(arg: &str, expected: usize, actual: usize) -> ParamError {
    common::rank_not_support(format!("{arg}.ndim = {actual}, {expected} expected"))
}

#[inline]
pub(crate) fn type_distinct(pairs: &[DigitLayout]) -> Result<DigitLayout, ParamError> {
    let [dt, tail @ ..] = pairs else {
        unreachable!("pairs empty");
    };
    if tail.iter().all(|it| it == dt) {
        Ok(*dt)
    } else {
        Err(type_mismatch(format!("{pairs:?} are not distinct")))
    }
}

#[inline]
pub(crate) fn dim_distinct(args: &[Argument<usize>]) -> Result<Argument<usize>, ParamError> {
    Argument::merge(args)
        .copied()
        .map_err(|_| shape_mismatch(format!("{args:?} are not distinct")))
}

macro_rules! op_trait {
    ($name:ident $($body:item)*) => {
        pub trait $name<H: common::Handle>:
            common::Operator<
            Handle = H,
            Args = Args<H>,
        >{$($body)*}
    };
}

pub(crate) use {get_static, op_trait};

#[cfg(test)]
pub(crate) use test_utils::*;

#[cfg(test)]
mod test_utils {
    use std::fmt;

    pub struct Diff {
        pub abs: f64,
        pub rel: f64,
    }

    impl Diff {
        pub fn new(a: f64, b: f64) -> Self {
            let abs = (a - b).abs();
            let rel = abs / (a.abs() + b.abs() + f64::EPSILON);
            Self { abs, rel }
        }
    }

    pub struct ErrorCollector {
        threshold: Diff,
        max_diff: Diff,
        outliers: Vec<usize>,
        count: usize,
    }

    impl ErrorCollector {
        pub fn new(abs: f64, rel: f64) -> Self {
            Self {
                threshold: Diff { abs, rel },
                max_diff: Diff { abs: 0.0, rel: 0.0 },
                outliers: vec![],
                count: 0,
            }
        }

        pub fn push(&mut self, diff: Diff) {
            self.max_diff.abs = f64::max(self.max_diff.abs, diff.abs);
            self.max_diff.rel = f64::max(self.max_diff.rel, diff.rel);

            if diff.abs > self.threshold.abs && diff.rel > self.threshold.rel {
                self.outliers.push(self.count);
            }

            self.count += 1;
        }

        pub fn summary(self) -> (usize, usize) {
            (self.outliers.len(), self.count)
        }
    }

    impl fmt::Display for ErrorCollector {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "abs: {:.3e}, rel: {:.3e}, outliers: {}/{}",
                self.max_diff.abs,
                self.max_diff.rel,
                self.outliers.len(),
                self.count,
            )
        }
    }
}
