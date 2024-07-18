use super::{
    args::{Meta, SampleArgs},
    Args, RandomSample,
};
use crate::nvidia_gpu::{dt_name, Handle as Gpu, Internal as Handle, EXPORT, EXPORT_H};
use common::{locate_error, ErrorPosition, QueueOf};
use cuda::{
    bindings::{CUresult, CUstream},
    AsRaw, DevByte, Stream,
};
use libloading::{Library, Symbol};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<Scheme>,
}

impl RandomSample<Gpu> for Operator {
    fn workspace(&self) -> usize {
        self.scheme.as_ref().expect("Scheme not set").workspace
    }
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
        if args.detail.is_argmax() {
            scheme.argmax(
                args.workspace.ptr,
                args.workspace.len,
                args.kv_pair_base,
                args.data_base,
                queue,
            )
        } else {
            scheme.sample(
                args.workspace.ptr,
                args.workspace.len,
                args.kv_pair_base,
                args.data_base,
                args.detail,
                queue,
            )
        }
    }
}

const SUCCESS: i32 = CUresult::CUDA_SUCCESS as _;
type ArgMaxFunc = unsafe extern "C" fn(
    *mut DevByte,   // workspace_ptr
    usize,          // workspace_len
    *mut DevByte,   // kv_pair
    *const DevByte, // data
    usize,          // n
    CUstream,       // stream
) -> i32;
type SampleFunc = unsafe extern "C" fn(
    *mut DevByte,   // workspace_ptr
    usize,          // workspace_len
    *mut DevByte,   // kv_pair
    *const DevByte, // data
    usize,          // n

    f32,   // random
    f32,   // temperature
    f32,   // topp
    usize, // topk

    CUstream,
) -> i32;

struct Scheme {
    meta: Meta,
    argmax: String,
    sample: String,
    workspace: usize,
    lib: Arc<Library>,
}

impl Scheme {
    fn new(meta: Meta, handle: &Arc<Handle>) -> Self {
        const CODE: &str = include_str!("sample.cuh");
        let dt = dt_name(meta.dt);
        let workspace_name = format!("workspace_{dt}");
        let argmax_name = format!("arg_max_{dt}");
        let sample_name = format!("random_sample_{dt}");
        let lib = handle.compile(
            format!("random_sample_{dt}"),
            handle.device().compute_capability(),
            || {
                format!(
                    r#"
{EXPORT_H}
{CODE}

{EXPORT}cudaError {workspace_name}(
    size_t *workspace_len, size_t n
) {{
    return random_sample_workspace<{dt}>(workspace_len, n);
}}

{EXPORT}cudaError {argmax_name}(
    void *workspace_ptr, size_t workspace_len,
    cub::KeyValuePair<int, {dt}> *kv_pair,
    {dt} *const data, size_t n,
    cudaStream_t stream
) {{
    return arg_max(
        workspace_ptr, workspace_len,
        kv_pair, data, n,
        stream);
}}

{EXPORT}cudaError {sample_name}(
    void *workspace_ptr, size_t workspace_len,
    cub::KeyValuePair<int, {dt}> *kv_pair,
    {dt} *const data, size_t n,

    float random,
    float temperature,
    float topp,
    size_t topk,

    cudaStream_t stream
) {{
    return random_sample(
        workspace_ptr, workspace_len,
        kv_pair, data, n,

        random,
        temperature,
        topp,
        topk,

        stream);
}}
"#
                )
            },
        );

        let mut ans = Self {
            meta,
            lib,
            argmax: argmax_name,
            sample: sample_name,
            workspace: 0,
        };
        ans.argmax.push('\0');
        ans.sample.push('\0');
        ans.fill_workspace(workspace_name);
        ans
    }

    fn fill_workspace(&mut self, mut workspace_name: String) {
        static CACHE: OnceLock<Mutex<HashMap<Meta, usize>>> = OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);

        if let Some(workspace) = cache.lock().unwrap().get(&self.meta) {
            self.workspace = *workspace;
        } else {
            type Func = unsafe extern "C" fn(*mut usize, usize) -> i32;
            workspace_name.push('\0');
            let func: Symbol<Func> = unsafe { self.lib.get(workspace_name.as_bytes()) }.unwrap();
            assert_eq!(SUCCESS, unsafe { func(&mut self.workspace, self.meta.n) });
        }
    }

    fn argmax(
        &self,
        workspace_ptr: *mut DevByte,
        workspace_len: usize,
        kv_pair: *mut DevByte,
        data: *const DevByte,
        stream: &Stream,
    ) -> Result<(), ErrorPosition> {
        if workspace_len < self.workspace {
            return Err(locate_error!("Workspace out of range"));
        }
        let func: Symbol<ArgMaxFunc> = unsafe { self.lib.get(self.argmax.as_bytes()) }.unwrap();
        let result = unsafe {
            func(
                workspace_ptr,
                workspace_len,
                kv_pair,
                data,
                self.meta.n,
                stream.as_raw(),
            )
        };
        if result == SUCCESS {
            Ok(())
        } else {
            Err(locate_error!("ArgMax failed with cuda error code {result}"))
        }
    }

    fn sample(
        &self,
        workspace_ptr: *mut DevByte,
        workspace_len: usize,
        kv_pair: *mut DevByte,
        data: *const DevByte,
        detail: SampleArgs,
        stream: &Stream,
    ) -> Result<(), ErrorPosition> {
        if workspace_len < self.workspace {
            return Err(locate_error!("Workspace out of range"));
        }
        let func: Symbol<SampleFunc> = unsafe { self.lib.get(self.sample.as_bytes()) }.unwrap();
        let result = unsafe {
            func(
                workspace_ptr,
                workspace_len,
                kv_pair,
                data,
                self.meta.n,
                rand::random(),
                detail.temperature,
                detail.top_p,
                detail.top_k,
                stream.as_raw(),
            )
        };
        if result == SUCCESS {
            Ok(())
        } else {
            Err(locate_error!("ArgMax failed with cuda error code {result}"))
        }
    }
}

#[test]
fn test() {
    use common::Operator as _;
    use digit_layout::types::F16;

    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let dev = cuda::Device::new(0);
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    <Operator as common::Operator>::scheme(&mut op, &Args::new(F16, 32000)).unwrap();
    println!("workspace = {}", op.scheme.as_ref().unwrap().workspace);
}
