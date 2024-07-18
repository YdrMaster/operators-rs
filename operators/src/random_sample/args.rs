use super::KVPair;
use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, ErrorPosition, Handle, TensorLayout, Workspace};
use digit_layout::{types::U32, DigitLayout};
use std::{
    hash::{Hash, Hasher},
    ptr::{null, null_mut},
};

pub struct Args<H: Handle> {
    pub kv_pair: TensorLayout,
    pub kv_pair_base: MutPtr<H>,
    pub data: TensorLayout,
    pub data_base: ConstPtr<H>,
    pub detail: SampleArgs,
    pub workspace: Workspace<H>,
}

#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl<H: Handle> Args<H> {
    pub fn new(dt: DigitLayout, n: usize) -> Self {
        Args {
            kv_pair: TensorLayout::new(KVPair::<()>::LAYOUT, [], []),
            kv_pair_base: null_mut(),
            data: TensorLayout::new(dt, [n.into()], [(dt.nbytes() as isize).into()]),
            data_base: null(),
            detail: SampleArgs {
                temperature: 0.0,
                top_p: 0.0,
                top_k: usize::MAX,
            },
            workspace: Workspace {
                ptr: null_mut(),
                len: 0,
            },
        }
    }
}

impl SampleArgs {
    #[inline]
    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Result<Self, ErrorPosition> {
        if temperature < 0.0 {
            return Err(locate_error!("Temperature must be non-negative"));
        }
        if top_k == 0 || top_p <= 0.0 {
            return Err(locate_error!("top_k and top_p must be positive"));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
        })
    }

    #[inline]
    pub fn is_argmax(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }
}

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
        if self.kv_pair.dt() != KVPair::<()>::LAYOUT {
            return Err(locate_error!("index must be KVpair"));
        }
        if self.data.dt().nbytes() > U32.nbytes() {
            return Err(locate_error!("data type too large"));
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
