use super::{args::Meta, Args, RmsNorm};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{algebraic, locate_error, ErrorPosition, QueueOf};
use cuda::Version;
use digit_layout::DigitLayout;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<(Scheme, Arc<ModuleBox>)>,
}

impl RmsNorm<Gpu> for Operator {}

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
        let Meta {
            dt_w,
            dt_a,
            n: _,
            d,
        } = args.meta()?;

        #[allow(unreachable_code, clippy::diverging_sub_expression)]
        let Some(&d) = d.get_static() else {
            self.scheme = Some((Scheme::Common { dt_w, dt_a }, todo!()));
        };

        let device = self.handle.device();
        let cc = device.compute_capability();
        let max_num_threads_block = device.block_limit().max_threads;

        if d <= max_num_threads_block {
            self.padding_scheme(dt_w, dt_a, d, cc)
        } else {
            self.folding_scheme(dt_w, dt_a, d, cc, max_num_threads_block, device.warp_size())
        }
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta { dt_w, dt_a, n, d } = args.meta()?;
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

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(nsy);
        get_or_err!(dsy);
        get_or_err!(nsx);
        get_or_err!(dsx);
        get_or_err!(dsw);

        let unit = algebraic!(dt_a)? as isize;
        if dsy != unit || dsx != unit || dsw != algebraic!(dt_w)? as isize {
            return Err(locate_error!("Unsupported layout"));
        };

        let (name, m, block_dims) = match self.scheme.as_ref() {
            Some((s, m)) => {
                if !s.is_match(dt_w, dt_a, d) {
                    return Err(locate_error!());
                }
                match s {
                    Scheme::Common { .. } => todo!(),
                    Scheme::Padding { name, .. } => (name, m, d),
                    Scheme::Folding {
                        name, block_size, ..
                    } => (name, m, *block_size),
                }
            }
            None => return Err(locate_error!("Scheme not set")),
        };

        let nsy = (nsy / unit) as i32;
        let nsx = (nsx / unit) as i32;
        let params = cuda::params![y_base, nsy, x_base, nsx, w_base, epsilon];

        m.launch(name, n as u32, block_dims as u32, params.as_ptr(), 0, queue);
        Ok(())
    }
}

enum Scheme {
    Common {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
    },
    Padding {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        name: CString,
    },
    Folding {
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        block_size: usize,
        name: CString,
    },
}

impl Scheme {
    fn is_match(&self, dt_w_: DigitLayout, dt_a_: DigitLayout, d_: usize) -> bool {
        match *self {
            Scheme::Common { dt_w, dt_a } => dt_w == dt_w_ && dt_a == dt_a_,
            Scheme::Padding { dt_w, dt_a, d, .. } => dt_w == dt_w_ && dt_a == dt_a_ && d == d_,
            Scheme::Folding { dt_w, dt_a, d, .. } => dt_w == dt_w_ && dt_a == dt_a_ && d == d_,
        }
    }
}

const CODE: &str = include_str!("rms_norm.cuh");

fn dt_name(digit_layout: DigitLayout) -> &'static str {
    use digit_layout::types as ty;
    match digit_layout {
        ty::F16 => "half",
        ty::F32 => "float",
        _ => unimplemented!(),
    }
}

impl Operator {
    fn padding_scheme(
        &mut self,
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        cc: Version,
    ) -> Result<(), ErrorPosition> {
        let ww = algebraic!(dt_w)? * 8;
        let wa = algebraic!(dt_a)? * 8;
        let name = format!("rms_norm_padding_w{ww}a{wa}_{d}");
        let tw = dt_name(dt_w);
        let ta = dt_name(dt_a);
        let module = self.handle.compile_kernel(&name, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    padding<{d}>
    (y, stride_y, x, stride_x, w, epsilon);
}}"#
            )
        });
        self.scheme = Some((
            Scheme::Padding {
                dt_w,
                dt_a,
                d,
                name: CString::new(name).unwrap(),
            },
            module,
        ));
        Ok(())
    }

    fn folding_scheme(
        &mut self,
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        d: usize,
        cc: Version,
        max_num_threads_block: usize,
        num_threads_warp: usize,
    ) -> Result<(), ErrorPosition> {
        if d % num_threads_warp != 0 {
            return Err(locate_error!());
        }
        let max_num_warp_block = max_num_threads_block / num_threads_warp;
        // num_warp_block in [1, max_num_warp_block]
        // num_threads_warp
        // num_items_thread in [1, 2, 4, 8] // 8 = 128bit / sizeof(half)
        // TODO 也许还能分得更好
        let to_divid = d / num_threads_warp;
        let num_warps_block = max_num_warp_block;
        let num_threads_block = num_threads_warp * num_warps_block;
        let num_items_thread = (to_divid + num_warps_block - 1) / num_warps_block;

        let ww = algebraic!(dt_w)? * 8;
        let wa = algebraic!(dt_a)? * 8;
        let name = format!("rms_norm_padding_w{ww}a{wa}_{num_threads_block}x{num_items_thread}");
        let tw = dt_name(dt_w);
        let ta = dt_name(dt_a);

        let module = self.handle.compile_kernel(&name, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    folding<{num_threads_block}, {num_items_thread}>
    (y, stride_y, x, stride_x, w, epsilon, {d});
}}"#
            )
        });

        self.scheme = Some((
            Scheme::Folding {
                dt_w,
                dt_a,
                d,
                block_size: num_threads_block,
                name: CString::new(name).unwrap(),
            },
            module,
        ));
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};

    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let dev = cuda::Device::new(0);
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);
    for k in 8..=13 {
        let d = 1 << k;
        op.scheme(&Args {
            y_layout: TensorLayout::new(F16, &[dyn_(), d.into()], &[dyn_(); 2]),
            y_base: null_mut(),
            x_layout: TensorLayout::new(F16, &[dyn_(), d.into()], &[dyn_(); 2]),
            x_base: null(),
            w_layout: TensorLayout::new(F16, &[d.into()], &[dyn_()]),
            w_base: null(),
            epsilon: 1e-5,
        })
        .unwrap();
        let (scheme, module) = op.scheme.as_ref().unwrap();
        match scheme {
            Scheme::Common { .. } => todo!(),
            Scheme::Padding { name, .. } => handle.apply(|ctx| {
                println!(
                    "{}\n{}",
                    name.to_str().unwrap(),
                    module.load(name, ctx).info()
                );
            }),
            Scheme::Folding { name, .. } => handle.apply(|ctx| {
                println!(
                    "{}\n{}",
                    name.to_str().unwrap(),
                    module.load(name, ctx).info()
                );
            }),
        }
    }
}
