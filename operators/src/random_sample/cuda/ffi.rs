use crate::{random_sample::SampleArgs, LaunchError};
use cuda::{bindings::CUstream, AsRaw, DevByte, Stream};
use libloading::Library;

type WorkspaceFunc = unsafe extern "C" fn(
    *mut usize, // argmax
    *mut usize, // random_sample
    usize,      // n
) -> i32;

type ArgMaxFunc = unsafe extern "C" fn(
    *mut DevByte,   // - kv_pair
    *const DevByte, //   logits
    usize,          //   n
    *mut DevByte,   // - workspace_ptr
    usize,          //   workspace_len
    CUstream,       //   stream
) -> i32;

type SampleFunc = unsafe extern "C" fn(
    *mut DevByte,   // - kv_pair
    *const DevByte, //   logits
    *const DevByte, //   indices
    usize,          //   n
    f32,            // - seed
    f32,            //   temperature
    f32,            //   topp
    usize,          //   topk
    *mut DevByte,   // - workspace_ptr
    usize,          //   workspace_len
    CUstream,       //   stream
) -> i32;

macro_rules! extern_c {
    ($ty:ty; $lib:expr, $name:expr; $($args:expr),* $(,)?) => {{
        let result = unsafe { $lib.get::<$ty>($name.as_bytes()).unwrap()( $( $args ),* ) };
        if result == ::cuda::bindings::CUresult::CUDA_SUCCESS as _ {
            Ok(())
        } else {
            Err($crate::execution_failed(format!(
                "{} failed with cuda error code {result}",
                $name
            )))
        }
    }};
}

pub(super) fn workspace_size(
    lib: &Library,
    name: &str,
    n: usize,
) -> Result<(usize, usize), LaunchError> {
    let mut argmax_size = 0;
    let mut sample_size = 0;
    extern_c!(WorkspaceFunc; lib, name; &mut argmax_size, &mut sample_size, n)?;
    Ok((argmax_size, sample_size))
}

pub(super) fn argmax(
    lib: &Library,
    name: &str,
    kv_pair: *mut DevByte,
    logits: *const DevByte,
    n: usize,
    workspace: &mut [DevByte],
    stream: &Stream,
) -> Result<(), LaunchError> {
    extern_c! { ArgMaxFunc;
        lib, name;

        kv_pair,
        logits,
        n,

        workspace.as_mut_ptr(),
        workspace.len(),
        stream.as_raw(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample(
    lib: &Library,
    name: &str,
    kv_pair: *mut DevByte,
    logits: *const DevByte,
    indices: *const DevByte,
    n: usize,
    config: SampleArgs,
    seed: f32,
    workspace: &mut [DevByte],
    stream: &Stream,
) -> Result<(), LaunchError> {
    extern_c! { SampleFunc;
        lib, name;

        kv_pair,
        logits,
        indices,
        n,

        seed,
        config.temperature,
        config.top_p,
        config.top_k,

        workspace.as_mut_ptr(),
        workspace.len(),
        stream.as_raw(),
    }
}

pub(super) fn format_code(
    dt: &str,
    workspace_name: &str,
    argmax_name: &str,
    sample_name: &str,
) -> String {
    use crate::cuda::{EXPORT, EXPORT_H};
    const CODE: &str = include_str!("sample.cuh");

    format!(
        r#"
{EXPORT_H}
{CODE}

{EXPORT}cudaError {workspace_name}(
    size_t *argmax,
    size_t *random_sample,
    size_t n
) {{
    return calculate_workspace_size<{dt}>(argmax, random_sample, n);
}}

{EXPORT}cudaError {argmax_name}(
    cub::KeyValuePair<int, {dt}> *kv_pair,
    {dt} const *logits,
    size_t n,

    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream
) {{
    return arg_max(
        kv_pair,
        logits,
        n,

        workspace_ptr,
        workspace_len,
        stream);
}}

{EXPORT}cudaError {sample_name}(
    cub::KeyValuePair<int, {dt}> *kv_pair,
    {dt} const *logits,
    unsigned int const *indices,
    size_t n,

    float random,
    float temperature,
    float topp,
    size_t topk,

    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream
) {{
    return random_sample(
        kv_pair,
        logits,
        indices,
        n,

        random,
        temperature,
        topp,
        topk,

        workspace_ptr,
        workspace_len,
        stream);
}}
"#
    )
}
