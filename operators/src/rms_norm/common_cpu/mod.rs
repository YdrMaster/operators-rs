use super::{args::Meta, Args, RmsNorm};
use crate::{common_cpu::Handle as Cpu, utils::get_or_err};
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
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
        let Meta { dt, n, d } = args.meta()?;
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

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(nsy);
        get_or_err!(dsy);
        get_or_err!(nsx);
        get_or_err!(dsx);
        get_or_err!(dsw);

        for i in 0..n as isize {
            let sum = (0..d as isize)
                .map(|j| unsafe { *x_base.offset(i * nsx + j * dsx).cast::<f16>() })
                .map(|x| x.to_f32().powi(2))
                .sum::<f32>();
            let k = f16::from_f32((sum / (d as f32) + epsilon).sqrt().recip());
            for j in 0..d as isize {
                unsafe {
                    let y = y_base.offset(i * nsy + j * dsy).cast::<f16>();
                    let x = x_base.offset(i * nsx + j * dsx).cast::<f16>();
                    let w = w_base.offset(j * dsw).cast::<f16>();
                    *y = k * *w * *x;
                }
            }
        }
        Ok(())
    }
}
