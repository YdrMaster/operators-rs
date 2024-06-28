mod handle;

pub mod fuesd_softmax;
pub mod mat_mul;
pub mod reform;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;

pub use common::*;
pub use record::{is_recording, start_record, stop_record};

#[cfg(use_cpu)]
pub use handle::common_cpu;

#[cfg(use_cuda)]
pub use handle::nvidia_gpu;
#[cfg(use_cuda)]
pub extern crate cublas;
#[cfg(use_cuda)]
pub extern crate cuda;

mod record;
mod utils;
