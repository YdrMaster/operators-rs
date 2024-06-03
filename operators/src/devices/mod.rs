#[cfg(detected_cpu)]
pub mod common_cpu;

#[cfg(detected_cuda)]
pub mod nvidia_gpu;
