use super::{args::Meta, Args, FusedSoftmax};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{algebraic, locate_error, ErrorPosition, QueueOf};
use cuda::Version;
use digit_layout::types::F16;
use std::{
    ffi::{c_float, CString},
    mem::size_of,
    sync::Arc,
};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<(Scheme, Arc<ModuleBox>)>,
}

impl FusedSoftmax<Gpu> for Operator {}

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
        let Meta { dt } = args.meta()?;
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

        let unit = algebraic!(dt)? as isize;
        if sa != unit {
            return Err(locate_error!("Unsupported layout"));
        };

        let Some((scheme, m)) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };

        let grid_dims = (nh as u32, seq_len as u32);
        let block_size = scheme.max_threads_block as u32;
        let sh = (sh / unit) as i32;
        let ss = (ss / unit) as i32;
        let att_len = att_len as u32;
        let params = cuda::params![att_base, 0i32, sh, ss, att_len];

        if att_len <= block_size {
            m.launch(
                &scheme.padding,
                grid_dims,
                att_len,
                params.as_ptr(),
                0,
                queue,
            );
        } else {
            let num_items_thread = (att_len + block_size - 1) / block_size;
            let smem = (num_items_thread * block_size) as usize;
            m.launch(
                &scheme.folding,
                grid_dims,
                block_size,
                params.as_ptr(),
                smem * size_of::<c_float>(),
                queue,
            );
        }

        Ok(())
    }
}

struct Scheme {
    max_threads_block: usize,
    padding: CString,
    folding: CString,
}

const NAME: &str = "fused_softmax";
const CODE: &str = include_str!("fused_softmax.cuh");
impl Operator {
    fn scheme(&mut self, cc: Version) -> Result<(), ErrorPosition> {
        let mask = "AttentionCausualMask";
        let max_threads_block = self.handle.device().block_limit().max_threads;
        let padding = format!("fused_softmax_padding_{max_threads_block}");
        let folding = format!("fused_softmax_folding_{max_threads_block}");

        let module = self.handle.compile_kernel(NAME, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {padding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x
){{
    padding<{max_threads_block}>
    (att, {mask}(), stride_z, stride_y, stride_x);
}}

extern "C" __global__ void {folding}(
    half *__restrict__ att,
    int const stride_z,
    int const stride_y,
    int const stride_x,

    unsigned int const att_len
){{
    folding<{max_threads_block}>
    (att, {mask}(), att_len, stride_z, stride_y, stride_x);
}}
"#
            )
        });
        self.scheme = Some((
            Scheme {
                max_threads_block,
                padding: CString::new(padding).unwrap(),
                folding: CString::new(folding).unwrap(),
            },
            module,
        ));
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use std::ptr::null_mut;

    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let dev = cuda::Device::new(0);
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    <Operator as common::Operator>::scheme(
        &mut op,
        &Args {
            att_layout: TensorLayout::new(F16, &[dyn_(); 3], &[dyn_(); 3]),
            att_base: null_mut(),
        },
    )
    .unwrap();
    let (scheme, module) = op.scheme.as_ref().unwrap();
    handle.apply(|ctx| {
        println!("============================");
        println!("{}", scheme.padding.to_str().unwrap());
        println!("{}", module.load(&scheme.padding, ctx).info());
        println!("{}", scheme.folding.to_str().unwrap());
        println!("{}", module.load(&scheme.folding, ctx).info());
    })
}
