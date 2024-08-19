use super::{args::Meta, Args, RmsNorm};
use crate::{common_cpu::Handle as Cpu, utils::get_or_err};
use common::{locate_error, ErrorPosition, QueueOf};
use half::f16;

pub struct Operator;

impl RmsNorm<Cpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let _meta = args.meta()?;
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        _queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
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

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(nsy);
        get_or_err!(dsy);
        get_or_err!(nsx);
        get_or_err!(dsx);
        get_or_err!(dsw);

        let attrs = Attributes {
            n,
            d,
            nsy,
            dsy,
            nsx,
            dsx,
            dsw,
            epsilon: *epsilon,
        };

        use digit_layout::types as ty;
        match (dt_w, dt_a) {
            (ty::F16, ty::F16) => Scheme {
                attrs,
                y: y_base.cast::<f16>(),
                x: x_base.cast::<f16>(),
                w: w_base.cast::<f16>(),
            }
            .calculate(),
            (ty::F32, ty::F16) => Scheme {
                attrs,
                y: y_base.cast::<f16>(),
                x: x_base.cast::<f16>(),
                w: w_base.cast::<f32>(),
            }
            .calculate(),
            (ty::F32, ty::F32) => Scheme {
                attrs,
                y: y_base.cast::<f32>(),
                x: x_base.cast::<f32>(),
                w: w_base.cast::<f32>(),
            }
            .calculate(),
            _ => unreachable!(),
        }

        Ok(())
    }
}

struct Attributes {
    n: usize,
    d: usize,
    nsy: isize,
    dsy: isize,
    nsx: isize,
    dsx: isize,
    dsw: isize,
    epsilon: f32,
}

impl Attributes {
    #[inline]
    fn n(&self) -> isize {
        self.n as _
    }
    #[inline]
    fn d(&self) -> isize {
        self.d as _
    }
    #[inline]
    fn k(&self, sum: f32) -> f32 {
        (sum / (self.d as f32) + self.epsilon).sqrt().recip()
    }
    #[inline]
    unsafe fn y<T>(&self, p: *mut T, i: isize, j: isize) -> *mut T {
        p.byte_offset(i * self.nsy + j * self.dsy)
    }
    #[inline]
    unsafe fn x<T>(&self, p: *const T, i: isize, j: isize) -> *const T {
        p.byte_offset(i * self.nsx + j * self.dsx)
    }
    #[inline]
    unsafe fn w<T>(&self, p: *const T, j: isize) -> *const T {
        p.byte_offset(j * self.dsw)
    }
}

struct Scheme<W, A> {
    attrs: Attributes,
    y: *mut A,
    x: *const A,
    w: *const W,
}

impl Scheme<f16, f16> {
    fn calculate(self) {
        let Self { attrs, y, x, w } = self;

        for i in 0..attrs.n() {
            let sum = (0..attrs.d())
                .map(|j| unsafe { *attrs.x(x, i, j) }.to_f32())
                .map(|x| x.powi(2))
                .sum::<f32>();
            let k = attrs.k(sum);
            for j in 0..attrs.d() {
                unsafe {
                    let y = attrs.y(y, i, j);
                    let x = attrs.x(x, i, j).read().to_f32();
                    let w = attrs.w(w, j).read().to_f32();
                    *y = f16::from_f32(k * w * x);
                }
            }
        }
    }
}

impl Scheme<f32, f16> {
    fn calculate(self) {
        let Self { attrs, y, x, w } = self;

        for i in 0..attrs.n() {
            let sum = (0..attrs.d())
                .map(|j| unsafe { *attrs.x(x, i, j) }.to_f32())
                .map(|x| x.powi(2))
                .sum::<f32>();
            let k = attrs.k(sum);
            for j in 0..attrs.d() {
                unsafe {
                    let y = attrs.y(y, i, j);
                    let x = attrs.x(x, i, j).read().to_f32();
                    let w = attrs.w(w, j).read();
                    *y = f16::from_f32(k * w * x);
                }
            }
        }
    }
}

impl Scheme<f32, f32> {
    fn calculate(self) {
        let Self { attrs, y, x, w } = self;

        for i in 0..attrs.n() {
            let sum = (0..attrs.d())
                .map(|j| unsafe { *attrs.x(x, i, j) })
                .map(|x| x.powi(2))
                .sum::<f32>();
            let k = attrs.k(sum);
            for j in 0..attrs.d() {
                unsafe {
                    let y = attrs.y(y, i, j);
                    let x = attrs.x(x, i, j).read();
                    let w = attrs.w(w, j).read();
                    *y = k * w * x;
                }
            }
        }
    }
}
