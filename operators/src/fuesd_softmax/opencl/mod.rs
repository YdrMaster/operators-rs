use super::{args::Meta, Args, FusedSoftmax};
use crate::{
    fuesd_softmax::args::AttnMask,
    get_static,
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    strides_not_support, ByteOf, LaunchError, QueueAlloc,
    SchemeDiversity::Low as LowDiversity,
    SchemeError,
};
use clrt::{
    bindings::{cl_int, cl_uint},
    Context,
};
use digit_layout::{types as Ty, DigitLayout};
use lru::LruCache;
use std::sync::Mutex;

pub struct Operator {
    ctx: Context,
    max_group_size: usize,
    schemes: Mutex<LruCache<DigitLayout, KernelCache>>,
}

/// block size = 512 x 8 -> context = 4k
const ITEMS_THREAD: usize = 8;

impl FusedSoftmax<ClDevice> for Operator {}

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
        let Meta { dt } = args.meta()?;
        self.cache_kernel(dt);
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt } = args.meta()?;
        self.cache_kernel(args.att_layout.dt());

        let Args {
            att_mask,
            att_layout,
            att_base,
        } = args;
        if !matches!(*att_mask, AttnMask::Causal) {
            todo!()
        }
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        let &[sh, ss, sa] = att_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nh seq_len att_len
            sh ss      sa
        }

        let unit = dt.nbytes() as isize;
        if sa != unit {
            return Err(strides_not_support("").into());
        };

        let group_size = last_power_of_two(att_len.min(self.max_group_size));
        let items_thread = att_len.div_ceil(group_size);

        let name = if items_thread <= ITEMS_THREAD {
            "softmax_register"
        } else {
            "softmax_global"
        };

        let mut softmax = self
            .schemes
            .lock()
            .unwrap()
            .get(&dt)
            .unwrap()
            .take(name)
            .unwrap();

        softmax
            .set_arg(0, att_base)
            .set_arg(1, seq_len as cl_uint)
            .set_arg(2, att_len as cl_uint)
            .set_arg(3, (sh / unit) as cl_int)
            .set_arg(4, (ss / unit) as cl_int)
            .launch(
                &[0, 0],
                &[group_size * seq_len, nh],
                &[group_size, 1],
                queue_alloc.queue(),
                None,
            );

        let mut cache = self.schemes.lock().unwrap();
        let program = cache.get(&dt).unwrap();
        program.put(name, softmax);

        Ok(())
    }
}

impl Operator {
    fn cache_kernel(&self, dt: DigitLayout) {
        self.schemes.lock().unwrap().get_or_insert(dt, || {
            let dt_a = match dt {
                Ty::F32 => "float",
                Ty::F16 => "half",
                _ => unimplemented!(),
            };
            let src = CodeGen::new(include_str!("fused_softmax.cl"))
                .define("Tval", dt_a)
                .define("ITEMS_THREAD", ITEMS_THREAD)
                .define("MASK", "causal_mask")
                .to_string();
            KernelCache::new(&self.ctx, &src, CL2_0)
        });
    }
}

#[inline(always)]
const fn last_power_of_two(n: usize) -> usize {
    1 << (usize::BITS - n.leading_zeros() - 1)
}

#[cfg(test)]
mod test {
    use super::{Args, AttnMask};
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::null_mut;
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]),
            att_base: null_mut(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        seq_len: usize,
        att_len: usize,
        att_base: *mut H::Byte,
    ) -> Args<H> {
        Args {
            att_mask: AttnMask::Causal,
            att_layout: TensorLayout::new_contiguous(dt, &[nh, seq_len, att_len]),
            att_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::{super::common_cpu::Operator as RefOp, Operator};
        use crate::{
            common_cpu::{Cpu, ThisThread},
            opencl::ClDevice,
            test_utils::{Diff, ErrorCollector},
            Operator as _,
        };
        use clrt::Platform;
        use digit_layout::types as ty;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
        use std::{iter::zip, time::Instant};

        let mut cpu_op = RefOp::new(&Cpu);
        for platform in Platform::all() {
            for device in platform.devices() {
                println!("device: {}", device.name());

                let context = device.context();
                let queue = context.queue();
                let mut cl_op = Operator::new(&ClDevice::new(context.clone(), Default::default()));
                cpu_op.scheme(&dyn_args(ty::F64), 0).unwrap();
                cl_op.scheme(&dyn_args(ty::F32), 0).unwrap();

                let nh = 32;
                for (seq_len, att_len) in [
                    (5, 5),
                    (1, 11),
                    (1, 12),
                    (1, 19),
                    (1, 20),
                    (1, 1023),
                    (1, 1024),
                    (1, 2048),
                    (7, 2048),
                    (7, 20443),
                    (7, 20480),
                ] {
                    let mut att = vec![0.0f64; nh * seq_len * att_len];
                    rand::rng().fill(&mut att[..]);
                    let mut att_svm = context.malloc::<f32>(nh * seq_len * att_len);
                    let mut map = queue.map_mut(&mut att_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(&mut *mem, &att) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let time = Instant::now();
                    cl_op
                        .launch(
                            &args(ty::F32, nh, seq_len, att_len, att_svm.as_mut_ptr().cast()),
                            &mut [],
                            &queue,
                        )
                        .unwrap();
                    queue.finish();
                    let cl_time = time.elapsed();
                    let time = Instant::now();
                    cpu_op
                        .launch(
                            &args(ty::F64, nh, seq_len, att_len, att.as_mut_ptr().cast()),
                            &mut [],
                            &ThisThread,
                        )
                        .unwrap();
                    let cpu_time = time.elapsed();
                    println!("cl: {cl_time:?} / cpu: {cpu_time:?}");

                    let map = queue.map(&mut att_svm);
                    let ([], mem, []) = (unsafe { map.align_to::<f32>() }) else {
                        panic!()
                    };

                    let diff = att
                        .into_par_iter()
                        .zip(mem)
                        .map(|(a, b)| Diff::new(a, *b as _))
                        .collect::<Vec<_>>();
                    queue.unmap(map);

                    let mut ec = ErrorCollector::new(f32::EPSILON as f64, 1e-3);
                    diff.into_iter().for_each(|diff| ec.push(diff));
                    println!("{ec}");

                    let (out, count) = ec.summary();
                    assert!(out * 1000 <= count);
                }
            }
        }
    }
}
