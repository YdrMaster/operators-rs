mod ffi;

use super::{
    args::{Meta, SampleArgs},
    Args, Indices, RandomSample,
};
use crate::{
    cuda::{dt_name, Gpu, Handle},
    get_static, strides_not_support, ByteOf, LaunchError, QueueAlloc, SchemeDiversity, SchemeError,
    Workspace,
};
use cuda::{DevByte, Stream};
use digit_layout::DigitLayout;
use ffi::format_code;
use libloading::Library;
use lru::LruCache;
use std::{
    alloc::Layout,
    num::NonZero,
    sync::{Arc, Mutex, OnceLock},
};

pub struct Operator {
    handle: Arc<Handle>,
    schemes: Mutex<LruCache<DigitLayout, Scheme>>,
}

impl RandomSample<Gpu> for Operator {
    fn build_indices<QA>(n: usize, queue_alloc: &QA) -> Indices<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut indices = queue_alloc.alloc(Layout::array::<u32>(n).unwrap().size());
        let mut host = vec![0u32; n];
        host.iter_mut().enumerate().for_each(|(i, x)| *x = i as u32);
        queue_alloc.queue().memcpy_h2d(&mut indices, &host);
        Indices { n, mem: indices }
    }
}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
            schemes: node.0.scheme_cache(SchemeDiversity::Low),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let meta = args.meta()?;
        let mut schemes = self.schemes.lock().unwrap();
        let scheme = schemes.get_or_insert(meta.dt, || Scheme::new(&self.handle, meta.dt));
        let Some(&n) = meta.n.get_static() else {
            return Ok(0);
        };
        let (argmax_size, sample_size) = scheme.workspace_size(n);
        drop(schemes);

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

        if dt.nbytes() as isize != sp {
            return Err(strides_not_support("").into());
        }
        if size_of::<u32>() as isize != si {
            return Err(strides_not_support("").into());
        }

        let scheme = self
            .schemes
            .lock()
            .unwrap()
            .get_or_insert(dt, || Scheme::new(&self.handle, dt))
            .clone();
        let (argmax_size, sample_size) = scheme.workspace_size(n);

        let stream = queue_alloc.queue();
        if args.config.is_argmax() {
            let mut workspace = Workspace::new(queue_alloc, workspace, argmax_size);
            scheme.argmax(*kv_pair_base, *logits_base, n, &mut workspace, stream)
        } else {
            let mut workspace = Workspace::new(queue_alloc, workspace, sample_size);
            scheme.sample(
                *kv_pair_base,
                *logits_base,
                *indices_base,
                n,
                *config,
                *seed,
                &mut workspace,
                stream,
            )
        }
    }
}

#[derive(Clone, Debug)]
struct Scheme {
    dt: DigitLayout,
    argmax: String,
    sample: String,
    lib: Arc<Library>,
}

impl Scheme {
    fn new(handle: &Arc<Handle>, dt: DigitLayout) -> Self {
        let t = dt_name(dt);
        let workspace_name = format!("workspace_{t}");
        let mut argmax_name = format!("arg_max_{t}");
        let mut sample_name = format!("random_sample_{t}");
        let lib = handle.compile(&sample_name, handle.device().compute_capability(), || {
            format_code(t, &workspace_name, &argmax_name, &sample_name)
        });

        argmax_name.push('\0');
        sample_name.push('\0');

        Self {
            dt,
            argmax: argmax_name,
            sample: sample_name,
            lib,
        }
    }

    fn workspace_size(&self, n: usize) -> (usize, usize) {
        type Key = (DigitLayout, u32);
        type Pair = (usize, usize);
        static CACHE: OnceLock<Mutex<LruCache<Key, Pair>>> = OnceLock::new();

        *CACHE
            .get_or_init(|| Mutex::new(LruCache::new(NonZero::new(16).unwrap())))
            .lock()
            .unwrap()
            .try_get_or_insert((self.dt, n as _), || {
                ffi::workspace_size(&self.lib, &format!("workspace_{}\0", dt_name(self.dt)), n)
            })
            .unwrap()
    }

    fn argmax(
        &self,
        kv_pair: *mut DevByte,
        logits: *const DevByte,
        n: usize,
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Result<(), LaunchError> {
        ffi::argmax(
            &self.lib,
            &self.argmax,
            kv_pair,
            logits,
            n,
            workspace,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        kv_pair: *mut DevByte,
        logits: *const DevByte,
        indices: *const DevByte,
        n: usize,
        config: SampleArgs,
        seed: f32,
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Result<(), LaunchError> {
        ffi::sample(
            &self.lib,
            &self.sample,
            kv_pair,
            logits,
            indices,
            n,
            config,
            seed,
            workspace,
            stream,
        )
    }
}

#[test]
fn test_compute() {
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        Operator as _,
    };
    use cuda::memcpy_d2h;
    use digit_layout::types as ty;
    use rand::Rng;
    use std::ptr::null;

    let Some(gpu) = Gpu::init() else {
        return;
    };

    let n = 32000;

    let cpu_op = RefOp::new(&Cpu);
    let mut gpu_op = Operator::new(&gpu);
    println!(
        "workspace = {}",
        gpu_op
            .scheme(&Args::layout(ty::F32, n), usize::MAX)
            .unwrap()
    );

    let mut logits = vec![0.0f32; n];
    rand::rng().fill(&mut logits[..]);

    // argmax
    {
        let kv_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let logits = rt.from_host(&logits);
            let mut kv = rt.malloc::<KVPair>(1);

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
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let logits = rt.from_host(&logits);
            let indices = Operator::build_indices(n, &stream).mem;
            let mut kv = rt.malloc::<KVPair>(1);

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
