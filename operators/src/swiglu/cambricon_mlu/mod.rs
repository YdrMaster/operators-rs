﻿use super::{args::Meta, Args, Swiglu};
use crate::{
    cambricon_mlu::{Handle as Mlu, Internal as Handle},
    utils::get_or_err,
};
use cndrv::AsRaw;
use cnnl::{cnnl, DataType, Tensor};
use common::{locate_error, ErrorPosition, QueueOf};
use digit_layout::types::F16;
use std::{ptr::null_mut, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    act: cnnl::bindings::cnnlActivationDescriptor_t,
    glu: cnnl::bindings::cnnlBiasActivationGluDescriptor_t,
}

impl Swiglu<Mlu> for Operator {}

impl common::Operator for Operator {
    type Handle = Mlu;
    type Args = Args<Mlu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
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
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let _meta = args.meta()?;
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
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

        let unit = dt.nbytes() as isize;
        if sgd != unit || sud != unit {
            return Err(locate_error!("Unsupported layout"));
        };

        let dt = DataType::from(dt);
        let gate = Tensor::new(dt, &[n as _, d as _], &[sgn as _, sgd as _]);
        let up = Tensor::new(dt, &[], &[]);

        let mut workspace_size = 0;

        self.handle.cnnl(queue, |handle| {
            cnnl!(cnnlGetConcatWorkspaceSize(
                handle.as_raw(),
                2,
                &mut workspace_size
            ));
            queue.ctx().malloc::<u8>(workspace_size);
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
