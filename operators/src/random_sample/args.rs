use super::KVPair;
use crate::{
    type_not_support, utils::rank_error, ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError,
    TensorLayout,
};
use digit_layout::{types as ty, DigitLayout};
use std::ptr::{null, null_mut};

pub struct Args<H: Hardware> {
    pub kv_pair: TensorLayout,
    pub kv_pair_base: MutPtr<H>,
    pub logits: TensorLayout,
    pub logits_base: ConstPtr<H>,
    pub indices: TensorLayout,
    pub indices_base: ConstPtr<H>,
    pub config: SampleArgs,
    pub seed: f32,
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

impl<H: Hardware> Args<H> {
    pub fn layout(dt: DigitLayout, n: usize) -> Self {
        Args {
            kv_pair: TensorLayout::new(KVPair::<()>::LAYOUT, &[], &[]),
            kv_pair_base: null_mut(),
            logits: TensorLayout::new(dt, &[n], &[dt.nbytes() as _]),
            logits_base: null(),
            indices: TensorLayout::new(ty::U32, &[n], &[ty::U32.nbytes() as _]),
            indices_base: null(),
            config: SampleArgs {
                temperature: 0.,
                top_p: 0.,
                top_k: usize::MAX,
            },
            seed: 0.,
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
        if temperature < 0. {
            return Err(SampleArgsError::NegativeTemperature);
        }
        if top_k == 0 || top_p <= 0. {
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
        self.temperature == 0. || self.top_k == 1
    }
}

#[derive(PartialEq, Eq, Debug)]
pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: MaybeDyn<usize>,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            kv_pair,
            logits,
            indices,
            ..
        } = self;

        if kv_pair.dt() != KVPair::<()>::LAYOUT {
            return Err(type_not_support("output must be KVpair"));
        }

        let dt_p = logits.dt();
        if dt_p.nbytes() > size_of::<u32>() {
            return Err(type_not_support("element too large"));
        }
        if indices.dt() != ty::U32 {
            return Err(type_not_support("indices must be u32"));
        }
        let &[n] = self.logits.shape() else {
            return Err(rank_error("logits", 1, self.logits.ndim()));
        };
        let &[_] = self.indices.shape() else {
            return Err(rank_error("indices", 1, self.indices.ndim()));
        };

        Ok(Meta { dt: dt_p, n })
    }
}
