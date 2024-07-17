use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, ErrorPosition, Handle, TensorLayout, Workspace};
use digit_layout::{layout, DigitLayout};
use std::{
    ffi::c_int,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{align_of, size_of},
};

pub struct Args<H: Handle> {
    pub kv_pair: TensorLayout,
    pub kv_pair_base: MutPtr<H>,
    pub data: TensorLayout,
    pub data_base: ConstPtr<H>,
    pub workspace: Workspace<H>,
}

#[repr(C)]
pub struct KVpair<T> {
    pub idx: c_int,
    pub val: c_int,
    pub _phantom: PhantomData<T>,
}

impl<T: Copy> KVpair<T> {
    pub fn new(idx: c_int, val: T) -> Self {
        const { assert!(size_of::<T>() <= size_of::<c_int>()) }
        const { assert!(align_of::<T>() <= align_of::<c_int>()) }

        let mut val_bytes = 0;
        let ptr = std::ptr::from_mut(&mut val_bytes).cast::<T>();
        unsafe { ptr.write(val) };

        Self {
            idx,
            val: val_bytes,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn into_raw(self) -> (c_int, c_int) {
        (self.idx, self.val)
    }
}

layout!(KV_PAIR u(64)x(2));

#[derive(PartialEq, Eq, Debug)]
pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: usize,
}

impl Hash for Meta {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dt.to_u32().hash(state);
        self.n.hash(state);
    }
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
