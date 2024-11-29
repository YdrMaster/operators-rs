use super::{args::Meta, Args, LayerNorm};
use crate::{common_cpu::Cpu, get_static, ByteOf, LaunchError, QueueAlloc, SchemeError};
use half::f16;
use num_traits::{real::Real, NumCast, ToPrimitive};
use std::ops::AddAssign;

pub struct Operator;

impl LayerNorm<Cpu> for Operator {}

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
        let Meta { dt_w, dt_a, n, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            scale_layout,
            scale_base,
            bias_layout,
            bias_base,
            epsilon,
        } = args;
        let &[nsy, dsy] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, dsx] = x_layout.strides() else {
            unreachable!()
        };
        let &[dss] = scale_layout.strides() else {
            unreachable!()
        };
        let &[dsb] = bias_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   d
            nsy dsy
            nsx dsx
                dss
                dsb
        }

        macro_rules! calculate {
            ($eps:expr; $w:ty, $a:ty) => {
                Scheme {
                    n,
                    d,
                    nsy,
                    dsy,
                    nsx,
                    dsx,
                    dss,
                    dsb,
                    epsilon: $eps,
                    y: y_base.cast::<$a>(),
                    x: x_base.cast::<$a>(),
                    s: scale_base.cast::<$w>(),
                    b: bias_base.cast::<$w>(),
                }
                .calculate()
            };
        }

        use digit_layout::types as ty;
        match (dt_w, dt_a) {
            (ty::F16, ty::F16) => calculate!(*epsilon       ; f16, f16),
            (ty::F32, ty::F16) => calculate!(*epsilon       ; f32, f16),
            (ty::F32, ty::F32) => calculate!(*epsilon       ; f32, f32),
            (ty::F64, ty::F64) => calculate!(*epsilon as f64; f64, f64),
            (_, _) => todo!(),
        }

        Ok(())
    }
}

struct Scheme<X, W, A> {
    n: usize,
    d: usize,
    nsy: isize,
    dsy: isize,
    nsx: isize,
    dsx: isize,
    dss: isize,
    dsb: isize,
    epsilon: X,
    y: *mut A,
    x: *const A,
    s: *const W,
    b: *const W,
}

impl<X, W, A> Scheme<X, W, A>
where
    X: Real + AddAssign,
    W: Real,
    A: Real,
{
    fn calculate(self) {
        for i in 0..self.n as isize {
            let mut sum = X::zero();
            let mut sum2 = X::zero();
            for j in 0..self.d as isize {
                let x: X = get(self.x, i * self.nsx + j * self.dsx);
                sum += x;
                sum2 += x * x;
            }
            let n = X::from(self.d).unwrap();
            let e = sum / n;
            let e2 = sum2 / n;
            let std = (e2 - e * e).sqrt();
            let k = (std + self.epsilon).recip();

            for j in 0..self.d as isize {
                let y = unsafe { &mut *self.y.byte_offset(i * self.nsy + j * self.dsy) };
                let x: X = get(self.x, i * self.nsx + j * self.dsx);
                let s: X = get(self.s, j * self.dss);
                let b: X = get(self.b, j * self.dsb);

                *y = A::from((x - e).mul_add(s * k, b)).unwrap();
            }
        }
    }
}

#[inline]
fn get<X: NumCast, T: ToPrimitive>(ptr: *const T, offset: isize) -> X {
    X::from(unsafe { ptr.byte_offset(offset).read() }).unwrap()
}
