use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, ErrorPosition, Handle, TensorLayout};
use digit_layout::{layout, DigitLayout};
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
};

pub struct Args<H: Handle> {
    pub kv_pair: TensorLayout,
    pub kv_pair_base: MutPtr<H>,
    pub data: TensorLayout,
    pub data_base: ConstPtr<H>,
}

#[repr(C)]
pub struct KVpair<T> {
    pub idx: u64,
    pub val: u64,
    pub _phantom: PhantomData<T>,
}

impl<T: Copy> KVpair<T> {
    pub fn new(idx: u64, val: T) -> Self {
        const { assert!(size_of::<T>() <= size_of::<u64>()) }
        const { assert!(align_of::<T>() <= align_of::<u64>()) }

        let mut val64 = 0u64;
        let ptr = std::ptr::from_mut(&mut val64).cast::<T>();
        unsafe { ptr.write(val) };

        Self {
            idx,
            val: val64,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn into_raw(self) -> (u64, u64) {
        (self.idx, self.val)
    }
}

layout!(KV_PAIR u(64)x(2));

#[derive(PartialEq, Eq, Debug)]
pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: usize,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        if self.kv_pair.dt() != KV_PAIR {
            return Err(locate_error!("index must be KVpair"));
        }
        let &[n] = self.data.shape() else {
            return Err(locate_error!());
        };
        let Some(&n) = n.get_static() else {
            return Err(locate_error!("n must be static"));
        };
        Ok(Meta {
            dt: self.data.dt(),
            n,
        })
    }
}
