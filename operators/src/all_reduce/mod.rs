#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_nccl)]
pub mod nccl;

mod args;
pub use args::Args;

crate::comm_trait!(AllReduce);
crate::non_comm!(NonAllReduce impl AllReduce);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    Mean,
}
