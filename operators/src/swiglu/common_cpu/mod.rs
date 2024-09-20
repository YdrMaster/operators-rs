use super::{args::Meta, Args, Swiglu};
use crate::{common_cpu::Handle as Cpu, utils::get_static};
use common::{LaunchError, QueueOf, SchemeError};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl Swiglu<Cpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), SchemeError> {
        let _meta = args.meta()?;
        Ok(())
    }

    fn launch(&self, args: &Self::Args, _queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
        let Meta { dt, n, d } = args.meta()?;
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[sgn, sgd] = gate_layout.strides() else {
            unreachable!()
        };
        let &[sun, sud] = up_layout.strides() else {
            unreachable!()
        };

        get_static! {
              n   d
            sgn sgd
            sun sud
        }

        macro_rules! calculate {
            ($ty:ty) => {
                Scheme::<$ty> {
                    n,
                    d,
                    sgn,
                    sgd,
                    sun,
                    sud,
                    gate_base: gate_base.cast(),
                    up_base: up_base.cast(),
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
    sgn: isize,
    sgd: isize,
    sun: isize,
    sud: isize,
    gate_base: *mut T,
    up_base: *const T,
}

unsafe impl<T> Send for Scheme<T> {}
unsafe impl<T> Sync for Scheme<T> {}

impl<T: Copy> Scheme<T> {
    fn loop_(&self, f: impl Sync + Fn(T, T) -> T) {
        for i in 0..self.n as isize {
            (0..self.d as isize).into_par_iter().for_each(|j| {
                let gate = unsafe { &mut *self.gate_base.byte_offset(i * self.sgn + j * self.sgd) };
                let up = unsafe { *self.up_base.byte_offset(i * self.sun + j * self.sud) };
                *gate = f(*gate, up);
            })
        }
    }
}

impl Scheme<f16> {
    #[inline]
    fn calculate(&self) {
        self.loop_(|gate, up| {
            let a = gate.to_f32();
            let b = up.to_f32();
            f16::from_f32(a * sigmoid_f32(a) * b)
        })
    }
}

impl Scheme<f32> {
    #[inline]
    fn calculate(&self) {
        self.loop_(|gate, up| gate * sigmoid_f32(gate) * up)
    }
}

impl Scheme<f64> {
    #[inline]
    fn calculate(&self) {
        self.loop_(|gate, up| gate * sigmoid_f64(gate) * up)
    }
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

#[inline(always)]
fn sigmoid_f64(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
