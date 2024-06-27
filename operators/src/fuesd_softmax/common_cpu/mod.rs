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
        let Meta { dt } = args.meta()?;
        let Args {
            att_layout,
            att_base,
        } = args;
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        let &[sh, ss, sa] = att_layout.strides() else {
            unreachable!()
        };

        if dt != F16 {
            return Err(locate_error!());
        }

        get_or_err!(nh);
        get_or_err!(seq_len);
        get_or_err!(att_len);
        get_or_err!(sh);
        get_or_err!(ss);
        get_or_err!(sa);

        let nh = nh as isize;
        let seq_len = seq_len as isize;
        let att_len = att_len as isize;

        for i in 0..nh {
            for j in 0..seq_len {
                let att = unsafe { att_base.offset(i * sh + j * ss) };
                let att = |k: isize| unsafe { &mut *att.offset(k * sa).cast::<f16>() };
                let causal = att_len - seq_len + j + 1;

                let max = (0..causal)
                    .map(att)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap()
                    .to_f32();

                let sum = (0..causal)
                    .map(att)
                    .map(|x| {
                        let exp = (x.to_f32() - max).exp();
                        *x = f16::from_f32(exp);
                        exp
                    })
                    .sum::<f32>();

                let div = f16::from_f32(sum.recip());
                (0..causal).map(att).for_each(|x| *x *= div);

                (causal..att_len).map(att).for_each(|x| *x = f16::ZERO);
            }
        }
        Ok(())
    }
}
