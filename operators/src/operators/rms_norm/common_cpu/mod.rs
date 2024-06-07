use super::{layout::SchemeLayout, KnTensorLayout, RmsNormScheme};
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

impl RmsNormScheme<Cpu> for Scheme {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn launch(
        &self,
        y: *mut <Cpu as Device>::Byte,
        x: *const <Cpu as Device>::Byte,
        w: *const <Cpu as Device>::Byte,
        epsilon: f32,
    ) {
        let SchemeLayout {
            n,
            d,
            stride_y,
            stride_x,
            offset_y,
            offset_x,
            offset_w,
        } = self.0;

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
