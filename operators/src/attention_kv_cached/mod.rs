mod args;
mod operator;

pub use args::Args;

crate::op_trait! { AttnKVCached
    fn workspace_size(&self) -> Option<usize>;
}

macro_rules! impl_op {
    ($dev:ident) => {
        pub mod $dev {
            pub type Operator = super::operator::Operator<
                crate::$dev::Handle,
                crate::rearrange::$dev::Operator,
                crate::attention::$dev::Operator,
            >;
        }
    };
}

#[cfg(use_cpu)]
impl_op!(common_cpu);

#[cfg(use_cuda)]
impl_op!(nvidia_gpu);
