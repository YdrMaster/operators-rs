mod args;
mod operator;

pub use args::Args;

crate::op_trait!(Attention);

macro_rules! impl_op {
    ($dev:ident, $proc:ident) => {
        pub type Operator = super::operator::Operator<
            crate::$dev::$proc,
            crate::mat_mul::$dev::Operator,
            crate::fuesd_softmax::$dev::Operator,
            crate::rearrange::$dev::Operator,
        >;
    };
}

#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod cuda;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_cl)]
pub mod opencl;
