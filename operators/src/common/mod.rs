mod error;
mod maybe_dyn;
mod pool;
mod tensor;
mod workspace;

pub use error::{functions::*, LaunchError, LaunchErrorKind, SchemeError, SchemeErrorKind};
pub use maybe_dyn::{dyn_, DynVal, MaybeDyn};
pub use pool::Pool;
pub use tensor::TensorLayout;
pub use traits::{ArgsOf, ByteOf, Hardware, Operator, QueueAlloc, QueueOf};

pub(crate) use maybe_dyn::{get_static, static_from};
pub(crate) use traits::{op_trait, ConstPtr, MutPtr};
pub(crate) use workspace::{Workspace, WorkspaceCollector};
#[allow(dead_code)]
pub(crate) enum SchemeDiversity {
    Low,
    Medium,
    High,
}

mod traits {
    use crate::{LaunchError, SchemeError};
    use dev_mempool::Alloc;
    use std::ops::DerefMut;

    /// 算力硬件抽象。
    ///
    /// 约定硬件如何存储和运行。
    /// 这个特质应该由管理硬件的基本单元的映射类型实现，通常是**硬件上下文**。
    pub trait Hardware {
        /// 硬件的存储单元类型。
        type Byte;
        /// 硬件的任务队列类型。
        type Queue<'ctx>;
    }

    pub type ByteOf<H> = <H as Hardware>::Byte;
    pub type QueueOf<'ctx, H> = <H as Hardware>::Queue<'ctx>;
    pub type ArgsOf<O> = <O as Operator>::Args;
    pub(crate) type MutPtr<H> = *mut <H as Hardware>::Byte;
    pub(crate) type ConstPtr<H> = *const <H as Hardware>::Byte;

    /// 绑定到队列的分配器。
    pub trait QueueAlloc: Alloc<Self::DevMem> {
        /// 队列分配器对应的硬件。
        type Hardware: Hardware;
        /// 分配器分配和回收的对象，表示对某块存储区域的所有权。
        type DevMem: DerefMut<Target = [ByteOf<Self::Hardware>]>;
        /// 分配器对应的队列。
        fn queue(&self) -> &QueueOf<Self::Hardware>;
    }

    /// 算子。
    pub trait Operator {
        /// 执行算子的硬件。
        type Hardware: Hardware;
        /// 算子的参数类型。
        type Args;

        /// 在指定硬件抽象上创建算子实例。
        fn new(processor: &Self::Hardware) -> Self;

        /// 规划执行方案。
        ///
        /// 通过向算子实例提供尽可能详细的参数来尽量确定算子执行方案。
        /// 通过允许参数中标量值、张量形状、张量步长和张量基址的动态性（[ArgVal] 或 [null](std::ptr::null)）来尽可能复用算子实例。
        ///
        /// 另外，需要传入一个最大工作空间容量。工作空间是与硬件存储单元相同类型的存储区域，供算子执行过程中使用。
        /// 规划执行方案时，将尽可能尝试计算一个满足最大工作空间容量的工作空间需求，作为返回值。
        ///
        /// 算子的返回值将保证不大于最大工作空间容量。如果算子还需要更多空间，可能产生运行时分配。
        ///
        /// 由于参数提供可能不全，有时无法计算出具体的工作空间需求，算子将返回 0 作为工作空间需求，并在执行时再计算实际的需求。
        fn scheme(
            &mut self,
            args: &Self::Args,
            max_workspace_size: usize,
        ) -> Result<usize, SchemeError>;

        /// 发射算子到任务队列。
        ///
        /// 如果算子实际需要的工作空间大于通过参数提供的工作空间，将通过流分配器分配和释放工作空间。
        fn launch<QA>(
            &self,
            args: &Self::Args,
            workspace: &mut [ByteOf<Self::Hardware>],
            queue_alloc: &QA,
        ) -> Result<(), LaunchError>
        where
            QA: QueueAlloc<Hardware = Self::Hardware>;
    }

    macro_rules! op_trait {
        ($name:ident $($body:item)*) => {
            pub trait $name<H: $crate::Hardware>:
                $crate::Operator<
                Hardware = H,
                Args = Args<H>,
            >{$($body)*}
        };
    }

    pub(crate) use op_trait;
}

pub mod utils {
    use super::{
        rank_not_support, shape_mismatch, type_mismatch, type_not_support, MaybeDyn, SchemeError,
    };
    use digit_layout::DigitLayout;

    #[cfg(use_cuda)]
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
    pub(crate) fn sizeof(dt: DigitLayout) -> Result<usize, SchemeError> {
        dt.nbytes()
            .ok_or_else(|| type_not_support(format!("{dt} not supported")))
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
