mod blob;
mod calculator;
mod diversity;
mod error;
mod maybe_dyn;
mod pool;
mod tensor;
mod unsigned;
mod workspace;

pub use blob::Blob;
pub use calculator::OffsetCalculator;
pub use error::{functions::*, LaunchError, LaunchErrorKind, SchemeError, SchemeErrorKind};
pub use maybe_dyn::{dyn_, DynVal, MaybeDyn};
pub use pool::Pool;
pub use tensor::TensorLayout;
pub use unsigned::Unsigned;
pub use workspace::Workspace;

pub(crate) use diversity::{SchemeCacheSize, SchemeDiversity};
pub(crate) use maybe_dyn::{get_static, static_from};
pub(crate) use workspace::WorkspaceCollector;

pub mod utils {
    use super::{rank_not_support, shape_mismatch, type_mismatch, MaybeDyn, SchemeError};
    use digit_layout::DigitLayout;

    #[cfg(any(use_cuda, use_cl))]
    #[inline]
    pub(crate) const fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let rem = a % b;
            a = b;
            b = rem;
        }
        a
    }

    #[inline]
    pub(crate) fn type_distinct(pairs: &[DigitLayout]) -> Result<DigitLayout, SchemeError> {
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
    pub(crate) fn rank_error(arg: &str, expected: usize, actual: usize) -> SchemeError {
        rank_not_support(format!("{arg}.ndim = {actual}, {expected} expected"))
    }

    #[inline]
    pub(crate) fn dim_distinct(args: &[MaybeDyn<usize>]) -> Result<MaybeDyn<usize>, SchemeError> {
        MaybeDyn::merge(args)
            .copied()
            .map_err(|_| shape_mismatch(format!("{args:?} are not distinct")))
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_utils {
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
                max_diff: Diff { abs: 0., rel: 0. },
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

        pub fn outliers(&self) -> &[usize] {
            &self.outliers
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
