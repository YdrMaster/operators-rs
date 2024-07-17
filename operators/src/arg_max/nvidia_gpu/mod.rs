use super::{args::Meta, ArgMax, Args};
use crate::nvidia_gpu::{dt_name, Handle as Gpu, Internal as Handle, EXPORT, EXPORT_H};
use common::{locate_error, ErrorPosition, QueueOf};
use cuda::{
    bindings::{CUresult, CUstream},
    AsRaw, DevByte,
};
use libloading::{Library, Symbol};
use std::{
    collections::HashMap,
    os::raw::c_uint,
    ptr::{null, null_mut},
    sync::{Arc, Mutex, OnceLock},
};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<Scheme>,
}

impl ArgMax<Gpu> for Operator {}

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
        self.scheme = Some(Scheme::new(args.meta()?, &self.handle));
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let meta = args.meta()?;
        let Some(scheme) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };
        if scheme.meta != meta {
            return Err(locate_error!("Scheme meta mismatch"));
        }
        if args.workspace.len < scheme.workspace {
            return Err(locate_error!("Workspace out of range"));
        }
        let result = unsafe {
            scheme.func()(
                args.workspace.ptr,
                (&args.workspace.len as *const usize).cast_mut(),
                args.kv_pair_base,
                args.data_base,
                meta.n as _,
                queue.as_raw(),
            )
        };
        if SUCCESS == result {
            Ok(())
        } else {
            Err(locate_error!("ArgMax failed with cuda error: {result}"))
        }
    }
}

type Func = unsafe extern "C" fn(
    *mut DevByte,
    *mut usize,
    *mut DevByte,
    *const DevByte,
    c_uint,
    CUstream,
) -> i32;
const SUCCESS: i32 = CUresult::CUDA_SUCCESS as _;

struct Scheme {
    meta: Meta,
    name: String,
    workspace: usize,
    lib: Arc<Library>,
}

impl Scheme {
    fn new(meta: Meta, handle: &Arc<Handle>) -> Self {
        let dt = dt_name(meta.dt);
        let name = format!("argmax_{dt}");
        let lib = handle.compile(&name, handle.device().compute_capability(), || {
            format!(
                r#"
{EXPORT_H}
#include <cub/device/device_reduce.cuh>
#include <cuda_fp16.h>

{EXPORT}cudaError argmax_{dt}(
    void *temp_storage, size_t *temp_storage_bytes,
    cub::KeyValuePair<int, {dt}> *output, {dt} const *data, unsigned int n,
    cudaStream_t stream) {{
    return cub::DeviceReduce::ArgMax(
        temp_storage, *temp_storage_bytes,
        data, output, n,
        stream);
}}"#
            )
        });

        let mut name = name;
        name.push('\0');
        let mut scheme = Self {
            meta,
            name,
            lib,
            workspace: 0,
        };
        scheme.fill_workspace();
        scheme
    }

    fn fill_workspace(&mut self) {
        static CACHE: OnceLock<Mutex<HashMap<Meta, usize>>> = OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);

        if let Some(workspace) = cache.lock().unwrap().get(&self.meta) {
            self.workspace = *workspace;
        } else {
            assert_eq!(SUCCESS, unsafe {
                self.func()(
                    null_mut(),
                    &mut self.workspace,
                    null_mut(),
                    null(),
                    0,
                    null_mut(),
                )
            });
        }
    }

    #[inline(always)]
    fn func(&self) -> Symbol<Func> {
        unsafe { self.lib.get(self.name.as_bytes()) }.unwrap()
    }
}

#[test]
fn test() {
    use super::KV_PAIR;
    use common::{Operator as _, TensorLayout, Workspace};
    use digit_layout::types::F16;

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
            kv_pair: TensorLayout::new(KV_PAIR, &[], &[]),
            kv_pair_base: null_mut(),
            data: TensorLayout::new(F16, &[32000.into()], &[2.into()]),
            data_base: null(),
            workspace: Workspace {
                ptr: null_mut(),
                len: 0,
            },
        },
    )
    .unwrap();

    println!("workspace = {}", op.scheme.as_ref().unwrap().workspace);
}
