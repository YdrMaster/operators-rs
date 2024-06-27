use super::args::{Args, Meta};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{locate_error, ErrorPosition, QueueOf};
use cuda::Version;
use digit_layout::{types::F16, DigitLayout};
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<(Scheme, Arc<ModuleBox>)>,
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
        let Meta { dt, n: _, d } = args.meta()?;

        if dt != F16 {
            todo!()
        }
        #[allow(unreachable_code)]
        let Some(&d) = d.get_static() else {
            self.scheme = Some((Scheme::Common { dt }, todo!()));
        };

        let device = self.handle.device();
        let cc = device.compute_capability();
        let max_num_threads_block = device.block_limit().max_threads;

        if d <= max_num_threads_block {
            self.padding_scheme(dt, d, cc)
        } else {
            self.folding_scheme(dt, d, cc, max_num_threads_block, device.warp_size())
        }
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

        get_or_err!(n);
        get_or_err!(d);
        get_or_err!(nsy);
        get_or_err!(dsy);
        get_or_err!(nsx);
        get_or_err!(dsx);
        get_or_err!(dsw);

        let unit = dt.nbytes() as isize;
        if dsy != unit || dsx != unit || dsw != unit {
            return Err(locate_error!("Unsupported layout"));
        };

        let (name, m, block_dims) = match self.scheme.as_ref() {
            Some((Scheme::Common { dt: scheme_dt }, _)) => {
                if dt != *scheme_dt {
                    return Err(locate_error!());
                }
                todo!()
            }
            Some((
                Scheme::Padding {
                    dt: scheme_dt,
                    d: scheme_d,
                    name,
                },
                m,
            )) => {
                if dt != *scheme_dt || d != *scheme_d {
                    return Err(locate_error!());
                }
                (name, m, d)
            }
            Some((
                Scheme::Folding {
                    dt: scheme_dt,
                    d: scheme_d,
                    block_size,
                    name,
                },
                m,
            )) => {
                if dt != *scheme_dt || d != *scheme_d {
                    return Err(locate_error!());
                }
                (name, m, *block_size)
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
        dt: DigitLayout,
    },
    Padding {
        dt: DigitLayout,
        d: usize,
        name: CString,
    },
    Folding {
        dt: DigitLayout,
        d: usize,
        block_size: usize,
        name: CString,
    },
}

const CODE: &str = include_str!("rms_norm.cuh");
impl Operator {
    fn padding_scheme(
        &mut self,
        dt: DigitLayout,
        d: usize,
        cc: Version,
    ) -> Result<(), ErrorPosition> {
        let name = format!("rms_norm_padding_f16_{d}");
        let module = self
            .handle
            .compile(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    half *__restrict__ y,
    int  const stride_y,
    half const *__restrict__ x,
    int  const stride_x,
    half const *__restrict__ w,
    float epsilon
){{
    padding<{d}>
    (y, stride_y, x, stride_x, w, epsilon);
}}"#
                )
            })
            .map_err(|(e, log)| locate_error!(format!("Failed to compile {name}: {e:?}\n{log}")))?;
        self.scheme = Some((
            Scheme::Padding {
                dt,
                d,
                name: CString::new(name).unwrap(),
            },
            module,
        ));
        Ok(())
    }

    fn folding_scheme(
        &mut self,
        dt: DigitLayout,
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

        let name = format!("rms_norm_folding_f16_{num_threads_block}x{num_items_thread}");
        let module = self
            .handle
            .compile(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    half *__restrict__ y,
    int  const stride_y,
    half const *__restrict__ x,
    int  const stride_x,
    half const *__restrict__ w,
    float epsilon
){{
    folding<{num_threads_block}, {num_items_thread}>
    (y, stride_y, x, stride_x, w, epsilon, {d});
}}"#
                )
            })
            .map_err(|(e, log)| locate_error!(format!("Failed to compile {name}: {e:?}\n{log}")))?;

        self.scheme = Some((
            Scheme::Folding {
                dt,
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
    use std::ptr::{null, null_mut};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
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
            Scheme::Common { dt: _ } => todo!(),
            Scheme::Padding { dt: _, d: _, name } => handle.apply(|ctx| {
                println!("{}", module.load(name, ctx).info());
            }),
            Scheme::Folding {
                dt: _,
                d: _,
                block_size: _,
                name,
            } => handle.apply(|ctx| {
                println!("{}", module.load(name, ctx).info());
            }),
        }
    }
}
