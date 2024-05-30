use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use half::f16;

use super::{layout::RmsNormSchemeLayout, RmsNorm, RmsNormScheme, RmsNormTensorLayout};
use crate::{
    devices::CommonCpu, locate_error, DataLayout, Device, ErrorPosition, Kernel, Operator, F16,
};

impl Operator for RmsNorm {
    type Device = CommonCpu;
    type Config = DataLayout;
    type ConfigError = ErrorPosition;

    fn config(conf: Self::Config) -> Result<Self, Self::ConfigError> {
        if conf == F16 {
            Ok(Self { _dt: F16 })
        } else {
            Err(locate_error!())
        }
    }

    type Kernel = RmsNormCpu;
    type LoadError = ();

    fn load(&self, _: &<Self::Device as Device>::Context) -> Result<Self::Kernel, Self::LoadError> {
        Ok(RmsNormCpu)
    }
}

pub struct RmsNormCpu;

impl Kernel<CommonCpu> for RmsNormCpu {
    type Scheme = RmsNormCpuScheme;
    type Config = RmsNormTensorLayout;
    type SchemeError = ErrorPosition;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError> {
        Ok(RmsNormCpuScheme(RmsNormSchemeLayout::new(F16, config)?))
    }
}

pub struct RmsNormCpuScheme(RmsNormSchemeLayout);

impl RmsNormScheme<CommonCpu> for RmsNormCpuScheme {
    fn launch(
        &self,
        y: *mut <CommonCpu as Device>::Byte,
        x: *const <CommonCpu as Device>::Byte,
        w: *const <CommonCpu as Device>::Byte,
        epsilon: f32,
    ) {
        let RmsNormSchemeLayout {
            n,
            d,
            stride_y,
            stride_x,
        } = self.0;

        let y = y.cast::<f16>();
        let x = x.cast::<f16>();
        let w = unsafe { from_raw_parts(w.cast::<f16>(), d) };

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
