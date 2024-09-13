#![allow(unused)]

use common::Handle;

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

macro_rules! get_or_err {
    ($($name:ident)*) => {
        $(
            let Some(&$name) = $name.get_static() else {
                return Err(common::locate_error!());
            };
        )*
    };
}

macro_rules! op_trait {
    ($name:ident $($body:item)*) => {
        pub trait $name<H: common::Handle>:
            common::Operator<
            Handle = H,
            Args = Args<H>,
            SchemeError = common::ErrorPosition,
            LaunchError = common::ErrorPosition,
        >{$($body)*}
    };
}

pub(crate) use {get_or_err, op_trait};

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
