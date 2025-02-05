//! ref: <https://zhuanlan.zhihu.com/p/264786866>

use super::{args::Meta, Args, Indices, KVPair, RandomSample};
use crate::{
    get_static,
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    strides_not_support, ByteOf, LaunchError, QueueAlloc,
    SchemeDiversity::Low as LowDiversity,
    SchemeError, Workspace,
};
use clrt::{bindings::cl_uint, Context};
use digit_layout::{types as Ty, DigitLayout};
use lru::LruCache;
use std::sync::Mutex;

pub struct Operator {
    ctx: Context,
    max_group_size: usize,
    schemes: Mutex<LruCache<SchemeKey, KernelCache>>,
}

impl RandomSample<ClDevice> for Operator {
    fn build_indices<QA>(_n: usize, _queue_alloc: &QA) -> Indices<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        Indices {
            n: 0,
            mem: _queue_alloc.alloc(0),
        }
    }
}

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(node: &Self::TopoNode) -> Self {
        let ctx = node.context().clone();
        let max_group_size = ctx
            .devices()
            .iter()
            .map(|d| d.max_group_size())
            .min()
            .unwrap()
            / 2; // 直接用最大 group 可能导致资源不足
        Self {
            ctx,
            max_group_size,
            schemes: node.new_cache(LowDiversity),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt, n } = args.meta()?;

        let Some(&n) = n.get_static() else {
            return Ok(0);
        };

        let key = self.cache_kernel(dt, n);
        let n_pairs = n / key.group_size / 2;

        Ok(match n_pairs {
            0 => unreachable!(),
            1 => 0,
            n => n * KVPair::<()>::LAYOUT.nbytes(),
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
        let Meta { dt, n } = args.meta()?;
        let &[s] = args.logits.strides() else {
            unreachable!()
        };
        if s.get_static().copied() != Some(dt.nbytes() as isize) {
            return Err(strides_not_support("").into());
        }

        get_static!(n);
        let Args {
            kv_pair_base,
            logits_base,
            config,
            ..
        } = args;

        if !config.is_argmax() {
            todo!()
        }

        let key = self.cache_kernel(dt, n);
        let n_pairs = n / key.group_size / 2;
        let reduce_size = last_power_of_two(n_pairs.min(self.max_group_size));

        let (mut build_pairs, mut reduce) = {
            let mut cache = self.schemes.lock().unwrap();
            let program = cache.get(&key).unwrap();
            let build_pairs = program.take("argmax_build_pairs").unwrap();
            let reduce = program.take("argmax_reduce").unwrap();
            (build_pairs, reduce)
        };

        if n_pairs == 1 {
            build_pairs
                .set_arg(0, logits_base)
                .set_arg(1, kv_pair_base)
                .set_arg(2, n as cl_uint)
                .set_arg(3, f32::NEG_INFINITY)
                .launch(
                    &[0],
                    &[key.group_size],
                    &[key.group_size],
                    queue_alloc.queue(),
                    None,
                );
        } else {
            let mut pairs = Workspace::new(
                queue_alloc,
                workspace,
                n_pairs * KVPair::<()>::LAYOUT.nbytes(),
            );
            build_pairs
                .set_arg(0, logits_base)
                .set_arg(1, pairs.as_mut_ptr())
                .set_arg(2, n as cl_uint)
                .set_arg(3, f32::NEG_INFINITY)
                .launch(
                    &[0],
                    &[n_pairs * key.group_size],
                    &[key.group_size],
                    queue_alloc.queue(),
                    None,
                );
            reduce
                .set_arg(0, pairs.as_ptr())
                .set_arg(1, kv_pair_base)
                .set_arg(2, n_pairs as cl_uint)
                .set_arg(3, f32::NEG_INFINITY)
                .launch(
                    &[0],
                    &[reduce_size],
                    &[reduce_size],
                    queue_alloc.queue(),
                    None,
                );
        }

        let mut cache = self.schemes.lock().unwrap();
        let program = cache.get(&key).unwrap();
        program.put("argmax_build_pairs", build_pairs);
        program.put("argmax_reduce", reduce);

        Ok(())
    }
}

impl Operator {
    fn cache_kernel(&self, dt: DigitLayout, n: usize) -> SchemeKey {
        // n = (global / group) x group x 2
        // - 每个线程至少处理 2 个元素；
        // - group_size 是不大于 n/2 且不大于 max_group_size 的 2 的幂；
        let group_size = last_power_of_two((n / 2).min(self.max_group_size));
        let key = SchemeKey { dt, group_size };
        self.schemes.lock().unwrap().get_or_insert(key, || {
            let dt = match dt {
                Ty::F32 => "float",
                Ty::F16 => "half",
                _ => unimplemented!(),
            };
            let src = CodeGen::new(include_str!("random_sample.cl"))
                .define("Tval", dt)
                .define("GROUP_SIZE", group_size)
                .to_string();
            KernelCache::new(&self.ctx, &src, CL2_0)
        });
        key
    }
}

#[inline(always)]
const fn last_power_of_two(n: usize) -> usize {
    1 << (usize::BITS - n.leading_zeros() - 1)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt: DigitLayout,
    group_size: usize,
}

#[test]
fn test_compute() {
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        opencl::ClDevice,
        Operator as _,
    };
    use clrt::Platform;
    use digit_layout::types as ty;
    use rand::seq::SliceRandom;
    use std::time::Instant;

    let mut logits = (0..32000).map(|x| x as f32).collect::<Vec<_>>();
    logits.shuffle(&mut rand::rng());

    let cpu_op = RefOp::new(&Cpu);
    for platform in Platform::all() {
        for device in platform.devices() {
            println!("device: {}", device.name());
            let context = device.context();
            let queue = context.queue();
            let cl_op = Operator::new(&ClDevice::new(context.clone(), Default::default()));

            let mut logits_svm = context.malloc::<f32>(logits.len());
            let mut kv_pair_svm = context.malloc::<KVPair>(1);

            let mut map = queue.map_mut(&mut logits_svm, false);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    logits.as_ptr(),
                    map.as_mut_ptr().cast(),
                    logits.len(),
                )
            };
            queue.unmap(map);

            let time = Instant::now();
            cl_op
                .launch(
                    &Args {
                        kv_pair_base: kv_pair_svm.as_mut_ptr().cast(),
                        logits_base: logits_svm.as_ptr().cast(),
                        ..Args::layout(ty::F32, logits.len())
                    },
                    &mut [],
                    &queue,
                )
                .unwrap();
            let map = queue.map(&mut kv_pair_svm);
            let kv_ans = unsafe { *map.as_ptr().cast::<KVPair<()>>() };
            queue.unmap(map);
            queue.finish();
            let cl_time = time.elapsed();

            let mut kv_ref = KVPair::new(u32::MAX, f32::MAX);
            let time = Instant::now();
            cpu_op
                .launch(
                    &Args {
                        kv_pair_base: (&raw mut kv_ref).cast(),
                        logits_base: logits.as_ptr().cast(),
                        ..Args::layout(ty::F32, logits.len())
                    },
                    &mut [],
                    &ThisThread,
                )
                .unwrap();
            let cpu_time = time.elapsed();

            println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
            assert_eq!(kv_ans.idx(), kv_ref.idx());
        }
    }
}
