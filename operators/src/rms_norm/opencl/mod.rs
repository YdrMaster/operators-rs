use super::{args::Meta, Args, RmsNorm};
use crate::{
    get_static,
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    ByteOf, LaunchError, QueueAlloc,
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
    schemes: Mutex<LruCache<SchemeKey, KernelCache>>,
}

impl RmsNorm<ClDevice> for Operator {}

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
        let Meta { dt_a, dt_w, d, .. } = args.meta()?;
        if let Some(&d) = d.get_static() {
            self.cache_kernel(dt_a, dt_w, d);
        }
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
        let Meta { dt_a, dt_w, n, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            w_base,
            epsilon,
            ..
        } = args;
        let &[nsy, ..] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, ..] = x_layout.strides() else {
            unreachable!()
        };
        get_static! {
            n d
            nsy
            nsx
        }

        let (key, group_size) = self.cache_kernel(dt_a, dt_w, d);

        let mut rms_norm = self
            .schemes
            .lock()
            .unwrap()
            .get(&key)
            .unwrap()
            .take("rms_norm")
            .unwrap();

        let unit = dt_a.nbytes() as isize;
        rms_norm
            .set_arg(0, y_base)
            .set_arg(1, (nsy / unit) as cl_int)
            .set_arg(2, x_base)
            .set_arg(3, (nsx / unit) as cl_int)
            .set_arg(4, w_base)
            .set_arg(5, epsilon)
            .set_arg(6, d as cl_uint)
            .launch(
                &[0],
                &[n * group_size],
                &[group_size],
                queue_alloc.queue(),
                None,
            );

        let mut cache = self.schemes.lock().unwrap();
        let program = cache.get(&key).unwrap();
        program.put("rms_norm", rms_norm);

        Ok(())
    }
}

impl Operator {
    fn cache_kernel(&self, dt_a: DigitLayout, dt_w: DigitLayout, d: usize) -> (SchemeKey, usize) {
        // group_size 是不大于 d 且不大于 max_group_size 的 2 的幂；
        let group_size = last_power_of_two(d.min(self.max_group_size));
        // 每线程可能处理多个数据
        let items_thread = d.div_ceil(group_size);
        let key = SchemeKey { dt_a, dt_w, d };
        self.schemes.lock().unwrap().get_or_insert(key, || {
            let dt_a = match dt_a {
                Ty::F32 => "float",
                Ty::F16 => "half",
                _ => unimplemented!(),
            };
            let dt_w = match dt_w {
                Ty::F32 => "float",
                Ty::F16 => "half",
                _ => unimplemented!(),
            };
            let src = CodeGen::new(include_str!("rms_norm.cl"))
                .define("Ta", dt_a)
                .define("Tw", dt_w)
                .define("ITEMS_THREAD", items_thread)
                .to_string();
            KernelCache::new(&self.ctx, &src, CL2_0)
        });
        (key, group_size)
    }
}

#[inline(always)]
const fn last_power_of_two(n: usize) -> usize {
    1 << (usize::BITS - n.leading_zeros() - 1)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt_a: DigitLayout,
    dt_w: DigitLayout,
    d: usize,
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    fn dyn_args<H: Hardware>(dt_w: DigitLayout, dt_a: DigitLayout, d: usize) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            y_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            y_base: null_mut(),
            x_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            x_base: null(),
            w_layout: TensorLayout::new_dyn(dt_w, &[d.into()], &[dyn_()]),
            w_base: null(),
            epsilon: 1e-5,
        }
    }

    fn args<H: Hardware>(
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        n: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        w_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt_a, &[n, d]);
        Args {
            y_layout: layout.clone(),
            y_base,
            x_layout: layout,
            x_base,
            w_layout: TensorLayout::new_contiguous(dt_w, &[d]),
            w_base,
            epsilon: 1e-5,
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

                for k in 2..=12 {
                    let n = 5;
                    let d = 1 << k;

                    cpu_op.scheme(&dyn_args(ty::F64, ty::F64, d), 0).unwrap();
                    cl_op.scheme(&dyn_args(ty::F32, ty::F32, d), 0).unwrap();

                    let mut x = vec![0.0f64; n * d];
                    let mut w = vec![0.0f64; d];
                    rand::rng().fill(&mut x[..]);
                    rand::rng().fill(&mut w[..]);

                    let mut x_svm = context.malloc::<f32>(n * d);
                    let mut w_svm = context.malloc::<f32>(d);

                    let mut map = queue.map_mut(&mut x_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(mem, &x) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let mut map = queue.map_mut(&mut w_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(mem, &w) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let mut y_svm = context.malloc::<f32>(n * d);
                    let time = Instant::now();
                    cl_op
                        .launch(
                            &args(
                                ty::F32,
                                ty::F32,
                                n,
                                d,
                                y_svm.as_mut_ptr().cast(),
                                x_svm.as_ptr().cast(),
                                w_svm.as_ptr().cast(),
                            ),
                            &mut [],
                            &queue,
                        )
                        .unwrap();
                    queue.finish();
                    let cl_time = time.elapsed();

                    //CPU
                    let mut y_ref = vec![0.; n * d];
                    let time = Instant::now();
                    cpu_op
                        .launch(
                            &args(
                                ty::F64,
                                ty::F64,
                                n,
                                d,
                                y_ref.as_mut_ptr().cast(),
                                x.as_ptr().cast(),
                                w.as_ptr().cast(),
                            ),
                            &mut [],
                            &ThisThread,
                        )
                        .unwrap();
                    let cpu_time = time.elapsed();

                    let map = queue.map(&mut y_svm);
                    let ([], y_ans, []) = (unsafe { map.align_to::<f32>() }) else {
                        panic!()
                    };

                    let diff = y_ref
                        .into_par_iter()
                        .zip(y_ans)
                        .map(|(a, b)| Diff::new(a, *b as _))
                        .collect::<Vec<_>>();
                    queue.unmap(map);

                    let mut ec = ErrorCollector::new(f32::EPSILON as f64, 1e-3);
                    diff.into_iter().for_each(|diff| ec.push(diff));
                    println!("{ec}");
                    println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
                    let (out, count) = ec.summary();
                    assert!(out * 1000 <= count);
                }
            }
        }
    }
}
