use super::{args::Meta, Args, RmsNorm};
use crate::{common_cpu::Cpu, get_static, ByteOf, LaunchError, QueueAlloc, SchemeError};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl RmsNorm<Cpu> for Operator {}

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
            w_layout,
            w_base,
            epsilon,
        } = args;
        let &[nsy, dsy] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, dsx] = x_layout.strides() else {
            unreachable!()
        };
        let &[dsw] = w_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   d
            nsy dsy
            nsx dsx
            dsw
        }

        macro_rules! calculate {
            ($w:ty, $a:ty) => {
                Scheme::<$w, $a> {
                    n,
                    d,
                    nsy,
                    dsy,
                    nsx,
                    dsx,
                    dsw,
                    epsilon: *epsilon,
                    y: y_base.cast(),
                    x: x_base.cast(),
                    w: w_base.cast(),
                }
                .calculate()
            };
        }

        use digit_layout::types as ty;
        match (dt_w, dt_a) {
            (ty::F16, ty::F16) => calculate!(f16, f16),
            (ty::F32, ty::F16) => calculate!(f32, f16),
            (ty::F32, ty::F32) => calculate!(f32, f32),
            (ty::F64, ty::F64) => calculate!(f64, f64),
            (_, _) => todo!(),
        }

        Ok(())
    }
}

struct Scheme<W, A> {
    n: usize,
    d: usize,
    nsy: isize,
    dsy: isize,
    nsx: isize,
    dsx: isize,
    dsw: isize,
    epsilon: f32,
    y: *mut A,
    x: *const A,
    w: *const W,
}

unsafe impl<W, A> Send for Scheme<W, A> {}
unsafe impl<W, A> Sync for Scheme<W, A> {}

impl<W, A> Scheme<W, A> {
    #[inline]
    unsafe fn y_ptr(&self, i: isize, j: isize) -> *mut A {
        self.y.byte_offset(i * self.nsy + j * self.dsy)
    }
    #[inline]
    unsafe fn x_ptr(&self, i: isize, j: isize) -> *const A {
        self.x.byte_offset(i * self.nsx + j * self.dsx)
    }
    #[inline]
    unsafe fn w_ptr(&self, j: isize) -> *const W {
        self.w.byte_offset(j * self.dsw)
    }
}

macro_rules! impl_k {
    ($ty:ty) => {
        fn k(&self, i: isize) -> $ty {
            let sum = (0..self.d as isize)
                .map(|j| unsafe { self.x(i, j) }.powi(2))
                .sum::<$ty>();
            (sum / (self.d as $ty) + self.epsilon as $ty).sqrt().recip()
        }
    };
}

impl<W> Scheme<W, f16> {
    impl_k!(f32);

    #[inline]
    unsafe fn y(&self, i: isize, j: isize, val: f32) {
        self.y_ptr(i, j).write(f16::from_f32(val))
    }
    #[inline]
    unsafe fn x(&self, i: isize, j: isize) -> f32 {
        self.x_ptr(i, j).read().to_f32()
    }
}
impl<W> Scheme<W, f32> {
    impl_k!(f32);

    #[inline]
    unsafe fn y(&self, i: isize, j: isize, val: f32) {
        self.y_ptr(i, j).write(val)
    }
    #[inline]
    unsafe fn x(&self, i: isize, j: isize) -> f32 {
        self.x_ptr(i, j).read()
    }
}
impl<W> Scheme<W, f64> {
    impl_k!(f64);

    #[inline]
    unsafe fn y(&self, i: isize, j: isize, val: f64) {
        self.y_ptr(i, j).write(val)
    }
    #[inline]
    unsafe fn x(&self, i: isize, j: isize) -> f64 {
        self.x_ptr(i, j).read()
    }
}

impl<A> Scheme<f16, A> {
    #[inline]
    unsafe fn w(&self, j: isize) -> f32 {
        self.w_ptr(j).read().to_f32()
    }
}
impl<A> Scheme<f32, A> {
    #[inline]
    unsafe fn w(&self, j: isize) -> f32 {
        self.w_ptr(j).read()
    }
}
impl<A> Scheme<f64, A> {
    #[inline]
    unsafe fn w(&self, j: isize) -> f64 {
        self.w_ptr(j).read()
    }
}

macro_rules! impl_scheme {
    ($w:ty, $a:ty) => {
        impl Scheme<$w, $a> {
            fn calculate(self) {
                for i in 0..self.n as isize {
                    let k = self.k(i);
                    (0..self.d as isize)
                        .into_par_iter()
                        .for_each(|j| unsafe { self.y(i, j, k * self.w(j) * self.x(i, j)) });
                }
            }
        }
    };
}

impl_scheme!(f16, f16);
impl_scheme!(f32, f16);
impl_scheme!(f32, f32);
impl_scheme!(f64, f64);
