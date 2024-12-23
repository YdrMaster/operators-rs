mod args;
mod operator;

pub use args::Args;

crate::op_trait!(Mlp);

macro_rules! impl_op {
    ($dev:ident, $proc:ident) => {
        pub type Operator = super::operator::Operator<
            crate::$dev::$proc,
            crate::mat_mul::$dev::Operator,
            crate::gelu::$dev::Operator,
            crate::rearrange::$dev::Operator,
        >;
    };
}

#[cfg(use_ascend)]
pub mod ascend;
#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
#[cfg(use_cl)]
pub mod opencl;
