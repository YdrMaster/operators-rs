#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;

mod args;
mod kv_pair;

pub use args::{Args, SampleArgs};
pub use kv_pair::KVPair;

crate::utils::op_trait! { RandomSample
    type Workspace<'ctx>: std::ops::DerefMut<Target = [H::Byte]>;
    fn workspace<'ctx>(&self, queue: &H::Queue<'ctx>) -> Self::Workspace<'ctx>;
}
