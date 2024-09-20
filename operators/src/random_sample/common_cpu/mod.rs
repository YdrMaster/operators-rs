use super::{Args, KVPair, RandomSample};
use crate::{
    between_f32::BetweenF32, common_cpu::Handle as Cpu, random_sample::args::SampleArgs,
    utils::sizeof,
};
use common::{strides_not_support, type_not_support, LaunchError, QueueOf, SchemeError};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{cmp::Ordering::Equal, slice::from_raw_parts};

pub struct Operator;

impl RandomSample<Cpu> for Operator {
    type Workspace<'ctx> = Vec<u8>;
    fn workspace<'ctx>(&self, _queue: &QueueOf<'ctx, Cpu>) -> Self::Workspace<'ctx> {
        vec![]
    }
}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, _args: &Self::Args) -> Result<(), SchemeError> {
        Ok(())
    }

    fn launch(&self, args: &Self::Args, _queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
        let meta = args.meta()?;
        let &[s] = args.data.strides() else {
            unreachable!()
        };
        let unit = sizeof(meta.dt)? as isize;
        if s.get_static().copied() != Some(unit) {
            return Err(strides_not_support("").into());
        }

        use digit_layout::types as ty;
        if args.detail.is_argmax() {
            macro_rules! argmax {
                ($ty:ty) => {
                    argmax::<$ty>(args.data_base, meta.n).into_raw()
                };
            }
            let kv = match meta.dt {
                ty::F16 => argmax!(f16),
                ty::F32 => argmax!(f32),
                e => return Err(type_not_support(format!("{e} not support")).into()),
            };
            unsafe { args.kv_pair_base.cast::<KVPair<()>>().write(kv) };
        } else {
            let SampleArgs {
                temperature,
                top_p,
                top_k,
            } = args.detail;
            macro_rules! random {
                ($ty:ty) => {
                    random::<$ty>(args.data_base, meta.n, temperature, top_p, top_k).into_raw()
                };
            }
            let kv = match meta.dt {
                ty::F16 => random!(f16),
                ty::F32 => random!(f32),
                e => return Err(type_not_support(format!("{e} not support")).into()),
            };
            unsafe { args.kv_pair_base.cast::<KVPair<()>>().write(kv) };
        }

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

fn random<T>(ptr: *const u8, len: usize, t: f32, top_p: f32, top_k: usize) -> KVPair<T>
where
    T: BetweenF32 + Copy,
{
    // sort
    let ptr = ptr as usize;
    let mut logits = (0..len)
        .into_par_iter()
        .map(|idx| KVPair::new(idx as _, unsafe { (ptr as *const T).add(idx).read() }.f32()))
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = logits[0].val();
    logits[0].set_val(1.);
    // softmax & sum
    for i in 1..logits.len() {
        let softmax = logits[i - 1].val() + ((logits[i].val() - max) / t).exp();
        logits[i].set_val(softmax);
    }
    // topk & topp & random
    let pk = logits[top_k.min(logits.len()) - 1].val();
    let pp = logits[logits.len() - 1].val() * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    let ans = *logits.iter().find(|p| p.val() >= plimit).unwrap();
    KVPair::new(ans.idx() as _, T::cast(ans.val()))
}
