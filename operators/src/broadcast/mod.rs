#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_nccl)]
pub mod nccl;

mod args;
pub use args::Args;

crate::comm_trait!(Broadcast);
crate::non_comm!(NonBroadcast impl Broadcast);
