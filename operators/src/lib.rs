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
pub use record::{is_recording, start_record, stop_record};

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

mod utils {
    macro_rules! op_trait {
        ($name:ident) => {
            pub trait $name<D: common::Device>:
                common::Scheme<Device = D, LayoutAttrs = LayoutAttrs, Params = Params<D>>
            {
            }
        };
    }

    pub(crate) use op_trait;
    pub(crate) type ConstPtr<D> = *const <D as common::Device>::Byte;
    pub(crate) type MutPtr<D> = *mut <D as common::Device>::Byte;
}

mod record;
