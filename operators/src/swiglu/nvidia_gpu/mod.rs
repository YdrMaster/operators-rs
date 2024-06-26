use super::args::{Args, Meta};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::{gcd, get_or_err},
};
use common::{locate_error, ErrorPosition, QueueOf};
use cuda::ComputeCapability;
use digit_layout::types::F16;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<Arc<ModuleBox>>,
}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let Meta { dt, n: _, d: _ } = args.meta()?;
        if dt != F16 {
            todo!()
        }
        self.scheme(self.handle.device().compute_capability())
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

        let Some(m) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };

        let sg = (sgn / unit) as i32;
        let su = (sun / unit) as i32;
        let params = cuda::params![gate_base, sg, up_base, su];

        let max_num_threads_block = self.handle.device().max_block_dims().0;
        let block = gcd(max_num_threads_block, d);

        m.launch(
            CString::new(NAME).unwrap(),
            (n as _, (d / block) as _),
            block as u32,
            params.as_ptr(),
            0,
            queue,
        );
        Ok(())
    }
}

const NAME: &str = "swiglu_f16";
const CODE: &str = include_str!("swiglu.cuh");
impl Operator {
    fn scheme(&mut self, cc: ComputeCapability) -> Result<(), ErrorPosition> {
        let module = self
            .handle
            .compile(&NAME, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {NAME}(
    half *__restrict__ gate,
    int const stride_gate,
    half const *__restrict__ up,
    int const stride_up
){{
    swiglu(gate, stride_gate, up, stride_up);
}}"#
                )
            })
            .map_err(|(e, log)| locate_error!(format!("Failed to compile {NAME}: {e:?}\n{log}")))?;
        self.scheme = Some(module);
        Ok(())
    }
}
