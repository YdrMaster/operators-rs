use super::{
    args::{Meta, SampleArgs},
    Args, RandomSample,
};
use crate::{
    get_static,
    nvidia_gpu::{dt_name, Gpu, Handle, EXPORT, EXPORT_H},
    scheme_not_compatible, scheme_not_set, strides_not_support,
    utils::sizeof,
    ByteOf, LaunchError, ParamError, QueueAlloc, SchemeError,
};
use dev_mempool::cuda::{bindings::CUstream, AsRaw, DevByte, Stream};
use digit_layout::DigitLayout;
use libloading::Library;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{
    alloc::Layout,
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

pub struct Operator {
    handle: Arc<Handle>,
    scheme: Option<Scheme>,
}

impl RandomSample<Gpu> for Operator {
    fn build_indices<QA>(n: usize, queue_alloc: &QA) -> QA::DevMem
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut indices = queue_alloc.alloc(Layout::array::<u32>(n).unwrap().size());
        let mut host = vec![0u32; n];
        host.par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i as u32);
        queue_alloc.queue().memcpy_h2d(&mut indices, &host);
        indices
    }
}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type Args = Args<Gpu>;

    fn new(processor: &Self::Hardware) -> Self {
        Self {
            handle: processor.0.clone(),
            scheme: None,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let scheme = Scheme::new(args.meta()?, &self.handle)?;
        let (argmax_size, sample_size) = (scheme.argmax.1, scheme.sample.1);
        self.scheme = Some(scheme);

        let (max, min) = if argmax_size > sample_size {
            (argmax_size, sample_size)
        } else {
            (sample_size, argmax_size)
        };
        Ok(if max <= max_workspace_size {
            max
        } else if min <= max_workspace_size {
            min
        } else {
            0
        })
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt, n, .. } = args.meta()?;
        let Args {
            kv_pair,
            kv_pair_base,
            logits,
            logits_base,
            indices,
            indices_base,
            config,
            seed,
        } = args;

        let [] = kv_pair.strides() else {
            unreachable!()
        };
        let &[sp] = logits.strides() else {
            unreachable!()
        };
        let &[si] = indices.strides() else {
            unreachable!()
        };

        get_static!(n sp si);

        let Some(scheme) = self.scheme.as_ref() else {
            return Err(scheme_not_set(""));
        };
        if scheme.dt != dt || scheme.n != n {
            return Err(scheme_not_compatible(""));
        }
        if sizeof(dt)? as isize != sp {
            return Err(strides_not_support("").into());
        }
        if size_of::<u32>() as isize != si {
            return Err(strides_not_support("").into());
        }

        let stream = queue_alloc.queue();
        if args.config.is_argmax() {
            if workspace.len() >= scheme.argmax.1 {
                scheme.argmax(*kv_pair_base, *logits_base, workspace, stream)
            } else {
                let mut workspace = queue_alloc.alloc(scheme.argmax.1);
                let res = scheme.argmax(*kv_pair_base, *logits_base, &mut workspace, stream);
                queue_alloc.free(workspace);
                res
            }
        } else if workspace.len() >= scheme.sample.1 {
            scheme.sample(
                *kv_pair_base,
                *logits_base,
                *indices_base,
                *config,
                *seed,
                workspace,
                stream,
            )
        } else {
            let mut workspace = queue_alloc.alloc(scheme.sample.1);
            let res = scheme.sample(
                *kv_pair_base,
                *logits_base,
                *indices_base,
                *config,
                *seed,
                &mut workspace,
                stream,
            );
            queue_alloc.free(workspace);
            res
        }
    }
}

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
        if result == $crate::cuda::bindings::CUresult::CUDA_SUCCESS as _ {
            Ok(())
        } else {
            Err($crate::execution_failed(format!(
                "{} failed with cuda error code {result}",
                $name
            )))
        }
    }};
}

struct Scheme {
    dt: DigitLayout,
    n: usize,
    argmax: (String, usize),
    sample: (String, usize),
    lib: Arc<Library>,
}

impl Scheme {
    fn new(meta: Meta, handle: &Arc<Handle>) -> Result<Self, ParamError> {
        const CODE: &str = include_str!("sample.cuh");
        let dt = dt_name(meta.dt);
        let mut workspace_name = format!("workspace_{dt}");
        let mut argmax_name = format!("arg_max_{dt}");
        let mut sample_name = format!("random_sample_{dt}");
        let lib = handle.compile(
            format!("random_sample_{dt}"),
            handle.device().compute_capability(),
            || {
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
            },
        );

        let n = meta.n;
        get_static!(n);

        argmax_name.push('\0');
        sample_name.push('\0');
        let (argmax_size, sample_size) = {
            type Key = (u32, u32);
            type WorkspaceSize = (usize, usize);
            static CACHE: OnceLock<Mutex<HashMap<Key, WorkspaceSize>>> = OnceLock::new();
            let cache = CACHE.get_or_init(Default::default);

            use std::collections::hash_map::Entry::*;
            match cache.lock().unwrap().entry((meta.dt.to_u32(), n as _)) {
                Occupied(entry) => *entry.get(),
                Vacant(entry) => {
                    workspace_name.push('\0');

                    let mut argmax_size = 0;
                    let mut sample_size = 0;
                    extern_c!(WorkspaceFunc; lib, workspace_name; &mut argmax_size, &mut sample_size, n).unwrap();
                    *entry.insert((argmax_size, sample_size))
                }
            }
        };
        Ok(Self {
            dt: meta.dt,
            n,
            argmax: (argmax_name, argmax_size),
            sample: (sample_name, sample_size),
            lib,
        })
    }

    fn argmax(
        &self,
        kv_pair: *mut DevByte,
        logits: *const DevByte,
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Result<(), LaunchError> {
        extern_c! { ArgMaxFunc;
            self.lib, self.argmax.0;

            kv_pair,
            logits,
            self.n,

            workspace.as_mut_ptr(),
            workspace.len(),
            stream.as_raw(),
        }
    }

    fn sample(
        &self,
        kv_pair: *mut DevByte,
        logits: *const DevByte,
        indices: *const DevByte,
        config: SampleArgs,
        seed: f32,
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Result<(), LaunchError> {
        extern_c! { SampleFunc;
            self.lib, self.sample.0;

            kv_pair,
            logits,
            indices,
            self.n,

            seed,
            config.temperature,
            config.top_p,
            config.top_k,

            workspace.as_mut_ptr(),
            workspace.len(),
            stream.as_raw(),
        }
    }
}

#[test]
fn test_compute() {
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        Operator as _,
    };
    use dev_mempool::cuda::memcpy_d2h;
    use digit_layout::types as ty;
    use rand::Rng;
    use std::ptr::null;

    let Some(gpu) = Gpu::init() else {
        return;
    };

    let n = 32000;

    let mut cpu_op = RefOp::new(&Cpu);
    let mut gpu_op = Operator::new(&gpu);
    cpu_op
        .scheme(&Args::layout(ty::F32, n), usize::MAX)
        .unwrap();
    println!(
        "workspace = {}",
        gpu_op
            .scheme(&Args::layout(ty::F32, n), usize::MAX)
            .unwrap()
    );

    let mut logits = vec![0.0f32; n];
    rand::thread_rng().fill(&mut logits[..]);

    // argmax
    {
        let kv_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();

            let logits = stream.from_host(&logits);
            let mut kv = stream.malloc::<KVPair>(1);

            gpu_op
                .launch(
                    &Args {
                        kv_pair_base: kv.as_mut_ptr(),
                        logits_base: logits.as_ptr().cast(),
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &stream,
                )
                .unwrap();

            let mut host = KVPair::new(u32::MAX, f32::MAX);
            memcpy_d2h(std::slice::from_mut(&mut host), &kv);
            host
        });

        let mut kv_ref = KVPair::new(u32::MAX, f32::MAX);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: (&mut kv_ref) as *mut _ as _,
                    logits_base: logits.as_ptr().cast(),
                    ..Args::layout(ty::F32, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        assert_eq!(kv_ans.idx(), kv_ref.idx());
    }

    // sample
    {
        let config = SampleArgs {
            temperature: 0.9,
            top_p: 0.9,
            top_k: 200,
        };
        let seed = 0.75;
        let kv_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();

            let logits = stream.from_host(&logits);
            let indices = Operator::build_indices(n, &stream);
            let mut kv = stream.malloc::<KVPair>(1);

            gpu_op
                .launch(
                    &Args {
                        kv_pair_base: kv.as_mut_ptr(),
                        logits_base: logits.as_ptr().cast(),
                        indices_base: indices.as_ptr().cast(),
                        config,
                        seed,
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &stream,
                )
                .unwrap();

            let mut host = KVPair::new(u32::MAX, f32::MAX);
            memcpy_d2h(std::slice::from_mut(&mut host), &kv);
            host
        });

        let mut kv_ref = KVPair::new(u32::MAX, f32::MAX);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: (&mut kv_ref) as *mut _ as _,
                    logits_base: logits.as_ptr().cast(),
                    indices_base: null(),
                    config,
                    seed,
                    ..Args::layout(ty::F32, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        assert_eq!(kv_ans.idx(), kv_ref.idx());
    }
}
