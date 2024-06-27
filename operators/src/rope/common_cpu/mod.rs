use super::{args::Meta, Args};
use crate::{common_cpu::Handle as Cpu, utils::get_or_err};
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
use half::f16;

pub struct Operator;

impl common::Operator for Operator {
    type Handle = Cpu;
    type Args = Args<Cpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(_handle: &Self::Handle) -> Self {
        Self
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let _meta = args.meta()?;
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        _queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta { dt, n } = args.meta()?;
        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
        } = args;
        let &[_, nh, dh] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(n);
        get_or_err!(nh);
        get_or_err!(dh);
        get_or_err!(st);
        get_or_err!(sh);
        get_or_err!(sd);
        get_or_err!(sp);

        let dh = dh as isize / 2;
        let sd = sd * 2;

        for i in 0..n as isize {
            let p = unsafe { *p_base.offset(i * sp).cast::<u32>() };
            for j in 0..nh as isize {
                for k in 0..dh {
                    let t = unsafe {
                        &mut *t_base.offset(i * st + j * sh + k * sd).cast::<(f16, f16)>()
                    };
                    let freq = p as f32 / theta.powf(k as f32 / dh as f32);
                    let (sin, cos) = freq.sin_cos();
                    let (a, b) = t;
                    let a_ = a.to_f32();
                    let b_ = b.to_f32();
                    *a = f16::from_f32(a_ * cos - b_ * sin);
                    *b = f16::from_f32(a_ * sin + b_ * cos);
                }
            }
        }
        Ok(())
    }
}
