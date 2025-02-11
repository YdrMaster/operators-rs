use super::{args::Meta, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    common_cpu::Cpu, get_static, strides_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
    Unsigned,
};
use digit_layout::{types as ty, DigitLayout};
use half::f16;

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
    ($a:ty) => {
        impl<T: Unsigned> Position<$a> for T {
            #[inline]
            fn freq_sin_cos(self, k: isize, dh: isize, theta: f32) -> ($a, $a) {
                (self.val() as $a / (theta as $a).powf(k as $a / dh as $a)).sin_cos()
            }
        }
    };
}

impl_position!(f32);
impl_position!(f64);

impl<A, P> Scheme<A, P>
where
    A: Activation,
    P: Position<A::Calculation> + Sync + Copy,
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
        } = self;
        let nt = nt as isize;
        let nh = nh as isize;
        let dh = dh as isize / 2;
        let sd = size_of::<[A; 2]>() as isize;

        for i in 0..nt {
            let t = unsafe { t_base.byte_offset(i * st).cast::<[A; 2]>() };
            let p = unsafe { *p_base.byte_offset(i * sp) };
            for j in 0..nh {
                for k in 0..dh {
                    let pair = unsafe { &mut *t.byte_offset(j * sh + k * sd) };
                    let (sin, cos) = p.freq_sin_cos(k, dh, theta);
                    A::calculate(pair, sin, cos)
                }
            }
        }
    }
}
