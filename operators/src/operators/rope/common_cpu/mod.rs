use super::{layout::SchemeLayout, KnTensorLayout, RopeScheme};
use crate::{
    devices::common_cpu::Device as Cpu, locate_error, DataLayout, Device, ErrorPosition, F16,
};
use half::f16;
use std::slice::{from_raw_parts, from_raw_parts_mut};

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

impl RopeScheme<Cpu> for Scheme {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn launch(&self, t: *mut <Cpu as Device>::Byte, pos: *const <Cpu as Device>::Byte, theta: f32) {
        let SchemeLayout {
            n,
            nh,
            dh,
            stride_token,
            stride_head,
            offset_t,
            offset_pos,
        } = self.0;
        let dh = dh / 2;
        let ts = stride_token / 2;
        let hs = stride_head / 2;

        let t = unsafe { t.add(offset_t) }.cast::<(f16, f16)>();
        let pos = unsafe { from_raw_parts(pos.add(offset_pos).cast::<u32>(), n) };

        for (i, pos) in pos.iter().enumerate() {
            let pos = *pos as f32;
            for j in 0..nh {
                let t = unsafe { t.offset(i as isize * ts + j as isize * hs) };
                let slice = unsafe { from_raw_parts_mut(t, dh) };
                for (k, slice) in slice.iter_mut().enumerate() {
                    let freq = pos / theta.powf(k as f32 / dh as f32);
                    let (sin, cos) = freq.sin_cos();
                    let (a, b) = slice;
                    let a_ = a.to_f32();
                    let b_ = b.to_f32();
                    *a = f16::from_f32(a_ * cos - b_ * sin);
                    *b = f16::from_f32(a_ * sin + b_ * cos);
                }
            }
        }
    }
}
