use super::{layout::SchemeLayout, LayoutAttrs, Params};
use crate::{
    devices::common_cpu::Device as Cpu, locate_error, DataLayout, ErrorPosition, QueueOf, F16,
};
use half::f16;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub struct Operator {
    dt: DataLayout,
}

impl crate::Operator<Cpu> for Operator {
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

impl crate::Scheme<Cpu, Operator> for Scheme {
    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    #[inline]
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        SchemeLayout::new(op.dt, layout).map(Self)
    }

    type Params<'ctx> = Params<Cpu>;
    fn launch(&self, params: &Self::Params<'_>, _queue: &QueueOf<Cpu>) {
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
        let &(t, pos, theta) = params;

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
