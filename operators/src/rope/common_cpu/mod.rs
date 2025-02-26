use super::{args::Meta, args::RopeType as R, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    common_cpu::Cpu, get_static, strides_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
    Unsigned,
};
use digit_layout::{types as ty, DigitLayout};
use half::f16;
use std::ptr::null;
#[derive(Copy, Clone)]
enum NtkPartsType {
    None,
    Yarn,
}

#[derive(Copy, Clone)]
enum SchemeType<T> {
    Rope {
        s: f32,
    },
    Long {
        long: *const T,
        short: *const T,
        s: f32,
        origin_pos: u32,
    },
    #[allow(dead_code)]
    NtkParts {
        alpha: f32,
        beta: f32,
        l0: f32,
        s: f32,
        ntktype: NtkPartsType,
    },
}
pub struct Operator;

impl Rope<Cpu> for Operator {
    fn build_sincos<QA>(
        _dt: DigitLayout,
        _nctx: usize,
        _dh: usize,
        queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        SinCosTable {
            nctx: 0,
            mem: queue_alloc.alloc(0),
        }
    }

    fn build_pos<I, QA>(
        dt: digit_layout::DigitLayout,
        nt: usize,
        iter: I,
        queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut blob = queue_alloc.alloc(dt.nbytes() * nt);
        match dt {
            ty::U32 => fill_pos(blob.as_mut_ptr().cast::<u32>(), nt, iter),
            ty::U64 => fill_pos(blob.as_mut_ptr().cast::<u64>(), nt, iter),
            _ => todo!(),
        }
        blob
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
        let Meta { dt_t, dt_p, nt, .. } = args.meta()?;
        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            rope_type,
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
        if sd != dt_t.nbytes() as isize {
            return Err(strides_not_support("").into());
        }

        match rope_type {
            R::Rope | R::Dyn { .. } | R::Ntk { .. } | R::Pi { .. } => {
                let (theta, s) = match rope_type {
                    R::Rope => (*theta, 1.),
                    R::Dyn { s, a } => (theta * (a * s - a + 1.), 1.),
                    R::Ntk { s } => (theta * s, 1.),
                    R::Pi { s } => (*theta, *s),
                    _ => unreachable!(),
                };
                macro_rules! calculate {
                    ($t:ty, $p:ty) => {
                        Scheme::<$t, $p> {
                            nt,
                            nh,
                            dh,
                            st,
                            sh,
                            sp,
                            theta,
                            t_base: t_base.cast(),
                            p_base: p_base.cast(),
                            scheme_type: SchemeType::Rope { s },
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
            }
            R::Long {
                long,
                short,
                max_pos,
                origin_pos,
            } => {
                let s = 1.0
                    + ((*max_pos as f32 / *origin_pos as f32).ln() / (*origin_pos as f32).ln())
                        .sqrt();
                macro_rules! calculate {
                    ($t:ty, $p:ty) => {
                        Scheme::<$t, $p> {
                            nt,
                            nh,
                            dh,
                            st,
                            sh,
                            sp,
                            theta: *theta,
                            t_base: t_base.cast(),
                            p_base: p_base.cast(),
                            scheme_type: SchemeType::Long {
                                long: long.cast(),
                                short: short.cast(),
                                s,
                                origin_pos: *origin_pos,
                            },
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
            }
            R::Yarn { alpha, beta, l0, s } | R::NtkParts { alpha, beta, l0, s } => {
                let ntktype = match rope_type {
                    R::NtkParts { .. } => NtkPartsType::None,
                    R::Yarn { .. } => NtkPartsType::Yarn,
                    _ => unreachable!(),
                };
                macro_rules! calculate {
                    ($t:ty, $p:ty) => {
                        Scheme::<$t, $p> {
                            nt,
                            nh,
                            dh,
                            st,
                            sh,
                            sp,
                            theta: *theta,
                            t_base: t_base.cast(),
                            p_base: p_base.cast(),
                            scheme_type: SchemeType::NtkParts {
                                alpha: *alpha,
                                beta: *beta,
                                l0: *l0,
                                s: *s,
                                ntktype,
                            },
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
            }
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
    sp: isize,
    theta: f32,
    t_base: *mut A,
    p_base: *const P,
    scheme_type: SchemeType<A>,
}

unsafe impl<A, P> Send for Scheme<A, P> {}
unsafe impl<A, P> Sync for Scheme<A, P> {}
/// 激活值。
trait Activation: Sized {
    /// 激活值类型决定计算类型。
    type Calculation: Copy;
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
    fn freq_sin_cos_rope(
        self,
        k: isize,
        dh: isize,
        theta: f32,
        s: f32,
    ) -> (Calculation, Calculation);
    fn freq_sin_cos_long(
        self,
        k: isize,
        dh: isize,
        t: f32,
        f: f32,
        s: f32,
    ) -> (Calculation, Calculation);
    #[allow(clippy::too_many_arguments)]
    fn freq_sin_cos_ntk_part(
        self,
        k: isize,
        dh: isize,
        theta: f32,
        alpha: f32,
        beta: f32,
        l0: f32,
        s: f32,
        ntktype: NtkPartsType,
    ) -> (Calculation, Calculation);
}

macro_rules! impl_position {
    ($a:ty) => {
        impl<T: Unsigned> Position<$a> for T {
            #[inline]
            fn freq_sin_cos_rope(self, k: isize, dh: isize, theta: f32, s: f32) -> ($a, $a) {
                (self.val() as $a * s as $a * (theta as $a).powf(k as $a / dh as $a).recip())
                    .sin_cos()
            }
            #[inline]
            fn freq_sin_cos_long(self, k: isize, dh: isize, t: f32, f: f32, s: f32) -> ($a, $a) {
                let (sin, cos) =
                    (self.val() as $a * (t as $a).powf(k as $a / dh as $a).recip() * (f as $a).recip() ).sin_cos();
                (sin * s as $a, cos * s as $a)
            }
            #[inline]
            fn freq_sin_cos_ntk_part(
                self,
                k: isize,
                dh: isize,
                theta: f32,
                alpha: f32,
                beta: f32,
                l0: f32,
                s: f32,
                ntktype: NtkPartsType,
            ) -> ($a, $a) {
                use std::f32::consts::PI;
                let pos = match ntktype {
                    NtkPartsType::None => self.val() as $a,
                    NtkPartsType::Yarn => self.val() as $a * (0.1 * s.ln() + 1.) as $a,
                };
                let theta = theta.powf(k as f32 / dh as f32).recip();
                let r = ((l0 / (2. * PI / theta) - alpha) / (beta - alpha)).clamp(0., 1.);
                (pos * ((1. - r) / s + r) as $a * theta as $a).sin_cos()
            }
        }
    };
}

impl_position!(f32);
impl_position!(f64);

impl<A, P> Scheme<A, P>
where
    A: Activation + Copy,
    P: Position<A::Calculation> + Sync + Copy + Unsigned,
{
    fn calculate(&self) {
        let &Self {
            nt,
            nh,
            dh,
            st,
            sh,
            sp,
            theta,
            t_base,
            p_base,
            scheme_type,
        } = self;
        let nt = nt as isize;
        let nh = nh as isize;
        let dh = dh as isize / 2;
        let sd = size_of::<[A; 2]>() as isize;

        for i in 0..nt {
            let t = unsafe { t_base.byte_offset(i * st).cast::<[A; 2]>() };
            let p = unsafe { *p_base.byte_offset(i * sp) };
            let factor = match scheme_type {
                SchemeType::Long {
                    long,
                    short,
                    origin_pos,
                    ..
                } => unsafe {
                    if p.val() < origin_pos as usize {
                        short
                    } else {
                        long
                    }
                },
                _ => null(),
            };
            for j in 0..nh {
                for k in 0..dh {
                    let pair = unsafe { &mut *t.byte_offset(j * sh + k * sd) };
                    let (sin, cos) = match scheme_type {
                        SchemeType::Rope { s } => p.freq_sin_cos_rope(k, dh, theta, s),
                        SchemeType::Long { s, .. } => {
                            //  TODO 这里先默认为 f32
                            let factor = unsafe { *factor.byte_offset(k * 4).cast() };
                            p.freq_sin_cos_long(k, dh, theta, factor, s)
                        }
                        SchemeType::NtkParts {
                            alpha,
                            beta,
                            l0,
                            s,
                            ntktype,
                        } => p.freq_sin_cos_ntk_part(k, dh, theta, alpha, beta, l0, s, ntktype),
                    };
                    A::calculate(pair, sin, cos)
                }
            }
        }
    }
}
