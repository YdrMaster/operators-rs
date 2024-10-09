use super::{args::Meta, Args, KVPair, RandomSample, SampleArgs};
use crate::{
    common_cpu::Cpu, get_static, strides_not_support, type_not_support, utils::sizeof, ByteOf,
    LaunchError, QueueAlloc, SchemeError,
};
use half::f16;
use std::{cmp::Ordering::Equal, slice::from_raw_parts};

pub struct Operator;

impl RandomSample<Cpu> for Operator {
    fn build_indices<QA>(_n: usize, queue_alloc: &QA) -> QA::DevMem
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        queue_alloc.alloc(0)
    }
}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

    fn new(_node: &Self::TopoNode) -> Self {
        Self
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt, n } = args.meta()?;
        let &[s] = args.logits.strides() else {
            unreachable!()
        };
        let unit = sizeof(dt)? as isize;
        if s.get_static().copied() != Some(unit) {
            return Err(strides_not_support("").into());
        }

        get_static!(n);

        use digit_layout::types as ty;
        let kv = if args.config.is_argmax() {
            macro_rules! argmax {
                ($ty:ty) => {
                    argmax::<$ty>(args.logits_base, n).into_raw()
                };
            }
            match dt {
                ty::F16 => argmax!(f16),
                ty::F32 => argmax!(f32),
                e => return Err(type_not_support(format!("{e} not support")).into()),
            }
        } else {
            let SampleArgs {
                temperature,
                top_p,
                top_k,
            } = args.config;
            macro_rules! random {
                ($ty:ty) => {
                    random::<$ty>(args.logits_base, n, temperature, top_p, top_k, args.seed)
                        .into_raw()
                };
            }
            match dt {
                ty::F16 => random!(f16),
                ty::F32 => random!(f32),
                e => return Err(type_not_support(format!("{e} not support")).into()),
            }
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

fn random<T>(ptr: *const u8, len: usize, t: f32, top_p: f32, top_k: usize, seed: f32) -> KVPair<T>
where
    T: BetweenF32 + Copy,
{
    // sort
    let ptr = ptr as usize;
    let mut logits = (0..len)
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
    let plimit = seed * f32::min(pk, pp);
    // sample
    let ans = *logits.iter().find(|p| p.val() >= plimit).unwrap();
    KVPair::new(ans.idx() as _, T::cast(ans.val()))
}

trait BetweenF32 {
    fn cast(f: f32) -> Self;
    fn f32(self) -> f32;
}

impl BetweenF32 for f32 {
    #[inline]
    fn cast(f: f32) -> Self {
        f
    }
    #[inline]
    fn f32(self) -> f32 {
        self
    }
}

impl BetweenF32 for half::f16 {
    #[inline]
    fn cast(f: f32) -> Self {
        Self::from_f32(f)
    }
    #[inline]
    fn f32(self) -> f32 {
        Self::to_f32(self)
    }
}
