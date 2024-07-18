use super::{ArgMax, Args, KVPair};
use crate::common_cpu::Handle as Cpu;
use common::{locate_error, ErrorPosition, QueueOf};
use half::f16;
use std::{cmp::Ordering::Equal, slice::from_raw_parts};

pub struct Operator;

impl ArgMax<Cpu> for Operator {
    #[inline]
    fn workspace(&self) -> usize {
        0
    }
}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), Self::SchemeError> {
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        _queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let meta = args.meta()?;
        let &[s] = args.data.strides() else {
            unreachable!()
        };
        if s.get_static().copied() != Some(meta.dt.nbytes() as _) {
            return Err(locate_error!());
        }
        use digit_layout::types as ty;
        let kv = match meta.dt {
            ty::F16 => argmax::<f16>(args.data_base, meta.n).into_raw(),
            ty::F32 => argmax::<f32>(args.data_base, meta.n).into_raw(),
            e => return Err(locate_error!("Unsupported data layout: {e:?}")),
        };
        unsafe { args.kv_pair_base.cast::<KVPair<()>>().write(kv) };
        Ok(())
    }
}

fn argmax<T: PartialOrd + Copy>(ptr: *const u8, len: usize) -> KVPair<T> {
    let (key, val) = unsafe { from_raw_parts(ptr.cast::<T>(), len) }
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
        .unwrap();
    KVPair::new(key as _, *val)
}
