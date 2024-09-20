use super::KVPair;
use crate::utils::{rank_not_support, sizeof, ConstPtr, MutPtr};
use common::{dyn_not_support, type_not_support, Handle, ParamError, TensorLayout};
use digit_layout::DigitLayout;
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

    pub workspace_size: usize,
    pub workspace: MutPtr<H>,
}

#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) top_k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleArgsError {
    NegativeTemperature,
    NonPositiveTop,
}

impl<H: Handle> Args<H> {
    pub fn new(dt: DigitLayout, n: usize) -> Self {
        Args {
            kv_pair: TensorLayout::new(KVPair::<()>::LAYOUT, &[], &[]),
            kv_pair_base: null_mut(),
            data: TensorLayout::new(dt, &[n], &[dt.nbytes().unwrap() as _]),
            data_base: null(),
            detail: SampleArgs {
                temperature: 0.0,
                top_p: 0.0,
                top_k: usize::MAX,
            },
            workspace_size: usize::MAX,
            workspace: null_mut(),
        }
    }
}

impl Default for SampleArgs {
    #[inline]
    fn default() -> Self {
        Self::ARG_MAX
    }
}

impl SampleArgs {
    pub const ARG_MAX: Self = Self {
        temperature: 0.,
        top_p: 1.,
        top_k: usize::MAX,
    };

    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Result<Self, SampleArgsError> {
        if temperature < 0.0 {
            return Err(SampleArgsError::NegativeTemperature);
        }
        if top_k == 0 || top_p <= 0.0 {
            return Err(SampleArgsError::NonPositiveTop);
        }
        Ok(Self {
            temperature,
            top_p: f32::min(top_p, 1.),
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
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        if self.kv_pair.dt() != KVPair::<()>::LAYOUT {
            return Err(type_not_support("index must be KVpair"));
        }

        let dt = self.data.dt();
        if sizeof(dt)? > size_of::<u32>() {
            return Err(type_not_support("element too large"));
        }
        let &[n] = self.data.shape() else {
            return Err(rank_not_support("logits", 1, self.data.ndim()));
        };

        Ok(Meta {
            dt,
            n: *n.get_static().ok_or_else(|| dyn_not_support(""))?,
        })
    }
}
