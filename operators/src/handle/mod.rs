#[cfg(use_ascend)]
pub mod ascend_card;
#[cfg(use_neuware)]
pub mod cambricon_mlu;
#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_cuda)]
pub mod nvidia_gpu;
