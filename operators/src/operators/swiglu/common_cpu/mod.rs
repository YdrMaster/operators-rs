use super::{layout::SchemeLayout, KnTensorLayout, SwigluScheme};
use crate::{
    devices::common_cpu::Device as Cpu, locate_error, DataLayout, Device, ErrorPosition, F16,
};
use half::f16;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Operator {
    _dt: DataLayout,
}

impl crate::Operator<Cpu> for Operator {
    type Config = DataLayout;
    type ConfigError = ErrorPosition;

    fn config(conf: Self::Config) -> Result<Self, Self::ConfigError> {
        if conf == F16 {
            Ok(Self { _dt: F16 })
        } else {
            Err(locate_error!())
        }
    }

    type Kernel = Kernel;
    type LoadError = ();

    fn load(&self, _: &<Cpu as Device>::Context) -> Result<Self::Kernel, Self::LoadError> {
        Ok(Kernel)
    }
}

pub struct Kernel;

impl crate::Kernel<Cpu> for Kernel {
    type Scheme = Scheme;
    type Config = KnTensorLayout;
    type SchemeError = ErrorPosition;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError> {
        Ok(Scheme(SchemeLayout::new(F16, config)?))
    }
}

pub struct Scheme(SchemeLayout);

impl SwigluScheme<Cpu> for Scheme {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn launch(&self, gate: *mut <Cpu as Device>::Byte, up: *const <Cpu as Device>::Byte) {
        let SchemeLayout {
            n,
            d,
            stride_gate,
            stride_up,
            offset_gate,
            offset_up,
        } = self.0;

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
