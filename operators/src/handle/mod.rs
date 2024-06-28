#[cfg(use_neuware)]
pub mod cambricon_mlu;
#[cfg(use_cpu)]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
