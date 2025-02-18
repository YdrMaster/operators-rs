use super::{
    args::{AttnMask, Meta},
    Args, FusedSoftmax,
};
use crate::{common_cpu::Cpu, get_static, ByteOf, LaunchError, QueueAlloc, SchemeError};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl FusedSoftmax<Cpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_node: &Self::TopoNode) -> Self {
        Self
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let _meta = args.meta()?;
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
        let Meta { dt } = args.meta()?;
        let Args {
            att_mask,
            att_layout,
            att_base,
        } = args;
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        let &[sh, ss, sa] = att_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nh seq_len att_len
            sh ss      sa
        }

        macro_rules! calculate {
            ($ty:ty) => {
                Scheme::<$ty> {
                    nh,
                    seq_len,
                    att_len,
                    sh,
                    ss,
                    sa,
                    att_base: att_base.cast(),
                }
                .calculate(*att_mask)
            };
        }

        use digit_layout::types as ty;
        match dt {
            ty::F16 => calculate!(f16),
            ty::F32 => calculate!(f32),
            ty::F64 => calculate!(f64),
            _ => todo!(),
        }
        Ok(())
    }
}

struct Scheme<T> {
    nh: usize,
    seq_len: usize,
    att_len: usize,
    sh: isize,
    ss: isize,
    sa: isize,
    att_base: *mut T,
}

unsafe impl<T> Send for Scheme<T> {}
unsafe impl<T> Sync for Scheme<T> {}

impl<T> Scheme<T> {
    fn loop_(&self, mask: AttnMask, f: impl Sync + Fn(isize, *mut T)) {
        let nh = self.nh as isize;
        let seq_len = self.seq_len as isize;
        let att_len = self.att_len as isize;

        (0..nh * seq_len).into_par_iter().for_each(|i| {
            let j = i / seq_len;
            let k = i % seq_len;
            let att = unsafe { self.att_base.byte_offset(j * self.sh + k * self.ss) };
            let causal = match mask {
                AttnMask::None => att_len,
                AttnMask::Causal => att_len - seq_len + k + 1,
            };
            f(causal, att)
        });
    }
}

impl Scheme<f16> {
    fn calculate(&self, mask: AttnMask) {
        let att_len = self.att_len as isize;
        self.loop_(mask, |causal, att| {
            let att = |k| unsafe { &mut *att.byte_offset(k * self.sa) };

            let max = (0..causal)
                .map(att)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap()
                .to_f32();

            let div = (0..causal)
                .map(att)
                .map(|x| {
                    let exp = (x.to_f32() - max).exp();
                    *x = f16::from_f32(exp);
                    exp
                })
                .sum::<f32>()
                .recip();

            (0..causal)
                .map(att)
                .for_each(|x| *x = f16::from_f32(x.to_f32() * div));
            (causal..att_len).map(att).for_each(|x| *x = f16::ZERO);
        });
    }
}

impl Scheme<f32> {
    fn calculate(&self, mask: AttnMask) {
        let att_len = self.att_len as isize;
        self.loop_(mask, |causal, att| {
            let att = |k| unsafe { &mut *att.byte_offset(k * self.sa) };

            let max = *(0..causal).map(att).max_by(|a, b| a.total_cmp(b)).unwrap();

            let div = (0..causal)
                .map(att)
                .map(|x| {
                    let exp = (*x - max).exp();
                    *x = exp;
                    exp
                })
                .sum::<f32>()
                .recip();

            (0..causal).map(att).for_each(|x| *x *= div);
            (causal..att_len).map(att).for_each(|x| *x = 0.);
        });
    }
}

impl Scheme<f64> {
    fn calculate(&self, mask: AttnMask) {
        let att_len = self.att_len as isize;
        self.loop_(mask, |causal, att| {
            let att = |k| unsafe { &mut *att.byte_offset(k * self.sa) };

            let max = *(0..causal).map(att).max_by(|a, b| a.total_cmp(b)).unwrap();

            let div = (0..causal)
                .map(att)
                .map(|x| {
                    let exp = (*x - max).exp();
                    *x = exp;
                    exp
                })
                .sum::<f64>()
                .recip();

            (0..causal).map(att).for_each(|x| *x *= div);
            (causal..att_len).map(att).for_each(|x| *x = 0.);
        });
    }
}
