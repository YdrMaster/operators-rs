use super::{args::Meta, Args, RmsNorm};
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
}

impl RmsNorm<Mlu> for Operator {}

impl common::Operator for Operator {
    type Handle = Mlu;
    type Args = Args<Mlu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
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

        let dt = DataType::from(dt);

        let mut op = null_mut();
        cnnl!(cnnlCreateFuseNormDescriptor(&mut op));
        cnnl!(cnnlSetFuseNormDescriptor(
            op,
            *epsilon,
            1.0,
            true,
            false,
            false,
            false,
            false,
            dt.as_raw(),
            cnnlTransformerNormType_t::CNNL_TRANSFORMER_RMSNORM,
        ));

        let y = Tensor::new(dt, &[n as _, d as _], &[nsy as _, dsy as _]);
        let x = Tensor::new(dt, &[n as _, d as _], &[nsx as _, dsx as _]);
        let w = Tensor::new(dt, &[d as _], &[dsw as _]);

        let mut workspace_size = 0;

        self.handle.cnnl(queue, |handle| {
            cnnl!(cnnlGetFuseNormWorkspaceSize(
                handle.as_raw(),
                op,
                x.as_raw(),
                &mut workspace_size,
            ));
            let mut workspace = queue.ctx().malloc::<u8>(workspace_size);
            cnnl!(cnnlFuseNorm(
                handle.as_raw(),
                op,
                x.as_raw(),
                x_base.cast(),
                w.as_raw(),
                w_base.cast(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                workspace.as_mut_ptr().cast(),
                workspace_size,
                y.as_raw(),
                y_base.cast(),
                null_mut(),
                null_mut()
            ));
        });

        cnnl!(cnnlDestroyFuseNormDescriptor(op));

        Ok(())
    }
}
