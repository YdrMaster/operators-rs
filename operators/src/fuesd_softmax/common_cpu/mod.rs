use super::{layout::SchemeLayout, FuesdSoftmax, LayoutAttrs, Params};
use common::{locate_error, ErrorPosition, QueueOf};
use dev_common_cpu::Device as Cpu;
use digit_layout::{types::F16, DigitLayout};
use half::f16;
use std::slice::from_raw_parts_mut;

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

impl FuesdSoftmax<Cpu> for Scheme {}

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
            nh,
            seq_len,
            att_len,
            stride_head,
            stride_token,
            offset,
        } = self.0;
        let &att = params;

        let att = unsafe { att.add(offset) }.cast::<f16>();
        let seq_len = seq_len as isize;
        let att_len = att_len as isize;

        for i in 0..nh as isize {
            for r in 0..seq_len {
                let ptr = unsafe { att.offset(i * stride_head + r * stride_token) };
                let slice = unsafe { from_raw_parts_mut(ptr, att_len as _) };
                let (att, tail) = slice.split_at_mut((att_len - seq_len + r + 1) as _);

                let max = att
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .to_f32();

                let sum = att
                    .iter_mut()
                    .map(|x| {
                        let exp = (x.to_f32() - max).exp();
                        *x = f16::from_f32(exp);
                        exp
                    })
                    .sum::<f32>();

                let div = f16::from_f32(sum.recip());
                att.iter_mut().for_each(|x| *x *= div);

                tail.fill(f16::ZERO);
            }
        }
    }
}
