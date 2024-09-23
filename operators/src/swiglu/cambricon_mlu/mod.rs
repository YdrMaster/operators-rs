use super::{args::Meta, Args, Swiglu};
use crate::{
    cambricon_mlu::{Handle as Mlu, Internal as Handle},
    utils::get_or_err,
};
use crate::{locate_error, QueueOf};
use cndrv::AsRaw;
use cnnl::{cnnl, DataType, Tensor};
use digit_layout::types::F16;
use std::{
    ptr::{addr_eq, null_mut},
    sync::Arc,
};

pub struct Operator {
    handle: Arc<Handle>,
    act: cnnl::bindings::cnnlActivationDescriptor_t,
    glu: cnnl::bindings::cnnlBiasActivationGluDescriptor_t,
}

impl Swiglu<Mlu> for Operator {}

impl crate::Operator for Operator {
    type Handle = Mlu;
    type Args = Args<Mlu>;

    fn new(handle: &Self::Handle) -> Self {
        let mut act = null_mut();
        cnnl!(cnnlCreateActivationDescriptor(&mut act));

        let mut glu = null_mut();
        cnnl!(cnnlCreateBiasActivationGluDescriptor(&mut glu));

        cnnl!(cnnlSetActivationDescriptor_v6(
            act,
            cnnlActivationMode_t::CNNL_ACTIVATION_SILU,
            cnnlActivationPreference_t::CNNL_ACTIVATION_HIGH_PRECISION,
            cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN,
            0.,
            0,
            0.,
            0.,
            true,
            true,
        ));
        cnnl!(cnnlSetBiasActivationGluDescriptor(
            glu,
            act,
            cnnlBiasActivationGluAlgo_t::CNNL_BIAS_ACTIVATION_GLU_ALGO_V2
        ));

        Self {
            handle: handle.0.clone(),
            act,
            glu,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), SchemeError> {
        args.meta().map(|_| ())
    }

    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
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

        let unit = dt.nbytes() as isize;
        if sgd != unit || sud != unit {
            return Err(locate_error!(
                "Unsupported layout: sgd != unit or sud != unit"
            ));
        };
        if sun != sgn {
            return Err(locate_error!("Unsupported layout: sun != sgn"));
        }
        if !addr_eq(unsafe { gate_base.add(d * unit as usize) }, up_base) {
            return Err(locate_error!("gate and up not continuous"));
        }

        let strides = [(sgn / unit) as _, 1];
        let dt = DataType::from(dt);
        let input = Tensor::new(dt, &[n as _, (2 * d) as _], &strides);
        let output = Tensor::new(dt, &[n as _, d as _], &strides);

        self.handle.cnnl(queue, |handle| {
            cnnl!(cnnlBiasActivationGluForward_v2(
                handle.as_raw(),
                self.glu,
                input.as_raw(),
                gate_base.cast(),
                null_mut(),
                null_mut(),
                output.as_raw(),
                gate_base.cast(),
            ));
        });

        Ok(())
    }
}

impl Drop for Operator {
    #[inline]
    fn drop(&mut self) {
        cnnl!(cnnlDestroyBiasActivationGluDescriptor(self.glu));
        cnnl!(cnnlDestroyActivationDescriptor(self.act));
    }
}
