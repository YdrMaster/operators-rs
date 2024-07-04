#[cfg(use_neuware)]
pub mod cambricon_mlu;
#[cfg(any(test, use_cpu))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
