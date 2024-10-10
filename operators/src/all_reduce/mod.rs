#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_nccl)]
pub mod nccl;

mod args;
pub use args::Args;

crate::comm_trait!(AllReduce);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    Mean,
}

mod non_all_reduce;
pub use non_all_reduce::NonAllReduce;
