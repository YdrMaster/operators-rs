use super::{args::Meta, Args, Rope};
use crate::{common_cpu::Handle as Cpu, utils::get_static};
use common::{LaunchError, QueueOf, SchemeError};
use half::f16;

pub struct Operator;

impl Rope<Cpu> for Operator {}

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
        let Meta { dt_t, dt_p, nt, .. } = args.meta()?;
        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            ..
        } = args;
        let &[_, nh, dh] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nt nh dh
            st sh sd
            sp
        }

        macro_rules! calculate {
            ($t:ty, $p:ty) => {
                Scheme::<$t, $p> {
                    nt,
                    nh,
                    dh,
                    st,
                    sh,
                    sd,
                    sp,
                    theta: *theta,
                    t_base: t_base.cast(),
                    p_base: p_base.cast(),
                }
                .calculate()
            };
        }

        use digit_layout::types as ty;
        match (dt_t, dt_p) {
            (ty::F16, ty::U32) => calculate!(f16, u32),
            (ty::F16, ty::U64) => calculate!(f16, u64),
            (ty::F32, ty::U32) => calculate!(f32, u32),
            (ty::F32, ty::U64) => calculate!(f32, u64),
            (ty::F64, ty::U32) => calculate!(f64, u32),
            (ty::F64, ty::U64) => calculate!(f64, u64),
            _ => todo!(),
        }
        Ok(())
    }
}

/// Calculate scheme.
/// A for activation, P for position.
struct Scheme<A, P> {
    nt: usize,
    nh: usize,
    dh: usize,
    st: isize,
    sh: isize,
    sd: isize,
    sp: isize,
    theta: f32,
    t_base: *mut A,
    p_base: *const P,
}

unsafe impl<A, P> Send for Scheme<A, P> {}
unsafe impl<A, P> Sync for Scheme<A, P> {}

/// 激活值。
trait Activation: Sized {
    /// 激活值类型决定计算类型。
    type Calculation;
    /// 计算流程。
    fn calculate(pair: &mut [Self; 2], sin: Self::Calculation, cos: Self::Calculation);
}

macro_rules! multilpy {
    ($a:expr, $b:expr, $sin:expr, $cos:expr) => {
        [$a * $cos - $b * $sin, $a * $sin + $b * $cos]
    };
}

impl Activation for f16 {
    type Calculation = f32;
    #[inline]
    fn calculate(pair: &mut [Self; 2], sin: Self::Calculation, cos: Self::Calculation) {
        let [a, b] = pair.map(f16::to_f32);
        *pair = multilpy!(a, b, sin, cos).map(f16::from_f32);
    }
}
impl Activation for f32 {
    type Calculation = Self;
    #[inline]
    fn calculate(pair: &mut [Self; 2], sin: Self::Calculation, cos: Self::Calculation) {
        let &mut [a, b] = pair;
        *pair = multilpy!(a, b, sin, cos)
    }
}
impl Activation for f64 {
    type Calculation = Self;
    #[inline]
    fn calculate(pair: &mut [Self; 2], sin: Self::Calculation, cos: Self::Calculation) {
        let &mut [a, b] = pair;
        *pair = multilpy!(a, b, sin, cos)
    }
}

trait Position<Calculation> {
    fn freq_sin_cos(self, k: isize, dh: isize, theta: f32) -> (Calculation, Calculation);
}

macro_rules! impl_position {
    ($p:ty,$a:ty) => {
        impl Position<$a> for $p {
            #[inline]
            fn freq_sin_cos(self, k: isize, dh: isize, theta: f32) -> ($a, $a) {
                (self as $a / (theta as $a).powf(k as $a / dh as $a)).sin_cos()
            }
        }
    };
}

impl_position!(u32, f32);
impl_position!(u32, f64);
impl_position!(u64, f32);
impl_position!(u64, f64);

impl<A, P> Scheme<A, P>
where
    A: Activation,
    P: Position<A::Calculation> + Sync + Copy,
{
    fn calculate(&self) {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let nt = self.nt as isize;
        let nh = self.nh as isize;
        let dh = self.dh as isize / 2;

        for i in 0..nt {
            let p = unsafe { *self.p_base.byte_offset(i * self.sp) };
            (0..dh).into_par_iter().for_each(|k| {
                for j in 0..nh {
                    let pair = unsafe {
                        &mut *self
                            .t_base
                            .byte_offset(i * self.st + j * self.sh + k * self.sd * 2)
                            .cast::<[A; 2]>()
                    };
                    let (sin, cos) = p.freq_sin_cos(k, dh, self.theta);
                    A::calculate(pair, sin, cos)
                }
            })
        }
    }
}
