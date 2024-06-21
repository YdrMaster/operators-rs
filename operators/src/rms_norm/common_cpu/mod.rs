use super::{layout::SchemeLayout, LayoutAttrs, Params, RmsNorm};
use common::{locate_error, ErrorPosition, QueueOf};
use dev_common_cpu::Device as Cpu;
use digit_layout::{types::F16, DigitLayout};
use half::f16;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Operator {
    dt: DigitLayout,
}

impl common::Operator for Operator {
    type Device = Cpu;

    type Config = DigitLayout;
    type Error = ErrorPosition;
    #[inline]
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        if *config == F16 {
            Ok(Self { dt: *config })
        } else {
            Err(locate_error!())
        }
    }
}

pub struct Scheme(SchemeLayout);

impl RmsNorm<Cpu> for Scheme {}

impl common::Scheme for Scheme {
    type Device = Cpu;
    type Operator = Operator;

    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    #[inline]
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        SchemeLayout::new(op.dt, layout).map(Self)
    }

    type Params = Params<Cpu>;
    fn launch(&self, params: &Self::Params, _queue: &QueueOf<Cpu>) {
        let SchemeLayout {
            n,
            d,
            stride_y,
            stride_x,
            offset_y,
            offset_x,
            offset_w,
        } = self.0;
        let &(y, x, w, epsilon) = params;

        let y = unsafe { y.add(offset_y) }.cast::<f16>();
        let x = unsafe { x.add(offset_x) }.cast::<f16>();
        let w = unsafe { from_raw_parts(w.add(offset_w).cast::<f16>(), d) };

        for i in 0..n as isize {
            let y = unsafe { from_raw_parts_mut(y.offset(stride_y * i), d) };
            let x = unsafe { from_raw_parts(x.offset(stride_x * i), d) };

            // (Σx^2 / d + δ)^(-1/2)
            let sum = x.iter().map(|x| x.to_f32()).map(|x| x * x).sum::<f32>();
            let k = f16::from_f32((sum / (d as f32) + epsilon).sqrt().recip());

            zip(y, zip(x, w)).for_each(|(y, (x, w))| *y = k * *w * *x);
        }
    }
}
