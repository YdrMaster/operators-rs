use super::{args::Meta, Args, Gelu};
use crate::{common_cpu::Cpu, get_static, ByteOf, LaunchError, QueueAlloc, SchemeError};
use half::f16;

pub struct Operator;

impl Gelu<Cpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

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
        let Meta { dt, n, d } = args.meta()?;
        let Args { layout, base } = args;
        let &[sn, sd] = layout.strides() else {
            unreachable!()
        };

        get_static! {
             n  d
            sn sd
        }

        macro_rules! calculate {
            ($ty:ty) => {
                Scheme::<$ty> {
                    n,
                    d,
                    sn,
                    sd,
                    base: base.cast(),
                }
                .calculate()
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
    n: usize,
    d: usize,
    sn: isize,
    sd: isize,
    base: *mut T,
}

unsafe impl<T> Send for Scheme<T> {}
unsafe impl<T> Sync for Scheme<T> {}

impl<T: Copy> Scheme<T> {
    fn loop_(&self, f: impl Sync + Fn(T) -> T) {
        for i in 0..self.n as isize {
            (0..self.d as isize).for_each(|j| {
                let data = unsafe { &mut *self.base.byte_offset(i * self.sn + j * self.sd) };
                *data = f(*data);
            })
        }
    }
}

impl Scheme<f16> {
    #[inline]
    fn calculate(&self) {
        self.loop_(|base| f16::from_f32(gelu_f32(base.to_f32())))
    }
}

impl Scheme<f32> {
    #[inline]
    fn calculate(&self) {
        self.loop_(gelu_f32)
    }
}

impl Scheme<f64> {
    #[inline]
    fn calculate(&self) {
        self.loop_(gelu_f64)
    }
}

#[inline(always)]
fn gelu_f32(x: f32) -> f32 {
    use std::f32::consts::FRAC_2_PI;
    0.5 * x * (1. + (FRAC_2_PI.sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

#[inline(always)]
fn gelu_f64(x: f64) -> f64 {
    use std::f64::consts::FRAC_2_PI;
    0.5 * x * (1. + (FRAC_2_PI.sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}
