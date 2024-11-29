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

#[cfg(use_ascend)]
pub mod ascend;
#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
#[cfg(use_cl)]
pub mod opencl;
