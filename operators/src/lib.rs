mod between_f32;
mod handle;

pub mod fuesd_softmax;
pub mod mat_mul;
pub mod random_sample;
pub mod rearrange;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;

pub mod attention;
pub mod attention_kv_cached;
pub mod mlp;

pub use common::*;
pub use record::{is_recording, start_record, stop_record};

#[cfg(use_cpu)]
pub use handle::common_cpu;

#[cfg(use_cuda)]
pub use handle::nvidia_gpu;
#[cfg(use_cuda)]
pub extern crate cublas;
#[cfg(use_cuda)]
pub use dev_mempool::cuda;
#[cfg(use_nccl)]
pub extern crate nccl;

#[cfg(use_neuware)]
pub use handle::cambricon_mlu;
#[cfg(use_neuware)]
pub extern crate cndrv;
#[cfg(use_neuware)]
pub extern crate cnnl;

#[cfg(use_ascend)]
pub use handle::ascend_card;
#[cfg(use_ascend)]
pub extern crate ascendcl;

mod record;
mod utils;
