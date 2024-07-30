mod args;
mod operator;

pub use args::Args;

crate::utils::op_trait!(Mlp);

macro_rules! impl_op {
    ($dev:ident) => {
        pub mod $dev {
            pub type Operator = super::operator::Operator<
                crate::$dev::Handle,
                crate::mat_mul::$dev::Operator,
                crate::swiglu::$dev::Operator,
            >;
        }
    };
}

#[cfg(use_cpu)]
impl_op!(common_cpu);

#[cfg(use_cuda)]
impl_op!(nvidia_gpu);
