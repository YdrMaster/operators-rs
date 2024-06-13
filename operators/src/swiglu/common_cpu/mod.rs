use super::{layout::SchemeLayout, LayoutAttrs, Params, Swiglu};
use common::{locate_error, DataLayout, ErrorPosition, QueueOf, F16};
use dev_common_cpu::Device as Cpu;
use half::f16;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Operator {
    dt: DataLayout,
}

impl common::Operator for Operator {
    type Device = Cpu;

    type Config = DataLayout;
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

impl Swiglu<Cpu> for Scheme {}

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
            stride_gate,
            stride_up,
            offset_gate,
            offset_up,
        } = self.0;
        let &(gate, up) = params;

        let gate = unsafe { gate.add(offset_gate).cast::<f16>() };
        let up = unsafe { up.add(offset_up).cast::<f16>() };

        for i in 0..n as isize {
            let gate = unsafe { from_raw_parts_mut(gate.offset(i * stride_gate), d) };
            let up = unsafe { from_raw_parts(up.offset(i * stride_up), d) };
            for (gate, up) in zip(gate, up) {
                let x = gate.to_f32();
                let y = up.to_f32();

                #[inline(always)]
                fn sigmoid(x: f32) -> f32 {
                    1. / (1. + (-x).exp())
                }

                *gate = f16::from_f32(x * sigmoid(x) * y);
            }
        }
    }
}
