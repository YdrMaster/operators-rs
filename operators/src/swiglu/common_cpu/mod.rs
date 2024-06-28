use super::{args::Meta, Args, Swiglu};
use crate::{common_cpu::Handle as Cpu, utils::get_or_err};
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
use half::f16;

pub struct Operator;

impl Swiglu<Cpu> for Operator {}

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
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[sgn, sgd] = gate_layout.strides() else {
            unreachable!()
        };
        let &[sun, sud] = up_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(sgn);
        get_or_err!(sgd);
        get_or_err!(sun);
        get_or_err!(sud);

        for i in 0..n as isize {
            for j in 0..d as isize {
                let gate = unsafe { &mut *gate_base.offset(i * sgn + j * sgd).cast::<f16>() };
                let up = unsafe { *up_base.offset(i * sun + j * sud).cast::<f16>() };

                let a = gate.to_f32();
                let b = up.to_f32();

                #[inline(always)]
                fn sigmoid(x: f32) -> f32 {
                    1. / (1. + (-x).exp())
                }

                *gate = f16::from_f32(a * sigmoid(a) * b);
            }
        }
        Ok(())
    }
}
