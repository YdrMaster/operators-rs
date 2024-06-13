#[cfg(use_cpu)]
pub extern crate dev_common_cpu as common_cpu;

#[cfg(use_cuda)]
pub extern crate dev_nvidia_gpu as nvidia_gpu;

#[cfg(use_neuware)]
pub extern crate dev_cambricon_mlu as cambricon_mlu;

pub mod fuesd_softmax;
pub mod mat_mul;
pub mod reform;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;

pub use common::*;

#[allow(unused)]
#[inline]
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}
