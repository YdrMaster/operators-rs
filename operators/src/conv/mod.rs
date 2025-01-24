mod args;
mod im2col;

pub use args::Args;

crate::op_trait!(Conv);

macro_rules! im2col {
    ($dev:ident, $proc:ident) => {
        pub type ConvIm2Col = super::im2col::Operator<
            crate::$dev::$proc,
            crate::rearrange::$dev::Operator,
            crate::mat_mul::$dev::Operator,
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
