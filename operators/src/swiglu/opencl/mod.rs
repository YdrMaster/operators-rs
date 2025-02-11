use super::{args::Meta, Args, Swiglu};
use crate::{
    get_static,
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    strides_not_support,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc,
    SchemeDiversity::Low as LowDiversity,
    SchemeError,
};
use clrt::{bindings::cl_int, Context};
use digit_layout::{types as Ty, DigitLayout};
use lru::LruCache;
use std::sync::Mutex;

pub struct Operator {
    ctx: Context,
    max_group_size: usize,
    schemes: Mutex<LruCache<SchemeKey, KernelCache>>,
}

impl Swiglu<ClDevice> for Operator {}

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
            / 2;
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
        let Meta { dt, d, .. } = args.meta()?;
        if let Some(&d) = d.get_static() {
            self.cache_kernel(dt, d);
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
        let Meta { dt, n, d } = args.meta()?;
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;
        let &[sgn, sgd] = gate_layout.strides() else {
            unreachable!()
        };
        let &[sun, sud] = up_layout.strides() else {
            unreachable!()
        };

        get_static! {
              n   d
            sgn sgd
            sun sud
        }

        let unit = dt.nbytes() as isize;
        if sgd != unit || sud != unit {
            return Err(strides_not_support("opencl: swiglu").into());
        };

        let sg = (sgn / unit) as i32;
        let su: i32 = (sun / unit) as i32;

        let (key, group_size) = self.cache_kernel(dt, d);

        let mut swiglu = self
            .schemes
            .lock()
            .unwrap()
            .get(&key)
            .unwrap()
            .take("swiglu")
            .unwrap();

        swiglu
            .set_arg(0, gate_base)
            .set_arg(1, (sg) as cl_int)
            .set_arg(2, up_base)
            .set_arg(3, (su) as cl_int)
            .launch(
                &[0, 0],
                &[n as usize, d as usize],
                &[1, group_size],
                queue_alloc.queue(),
                None,
            );

        let mut cache = self.schemes.lock().unwrap();
        let program = cache.get(&key).unwrap();
        program.put("swiglu", swiglu);
        Ok(())
    }
}

impl Operator {
    fn cache_kernel(&self, dt: DigitLayout, d: usize) -> (SchemeKey, usize) {
        // 求最大公约数以便均匀划分工作项
        let group_size = gcd(self.max_group_size, d);

        let key = SchemeKey { dt, d };
        self.schemes.lock().unwrap().get_or_insert(key, || {
            let dt = match dt {
                Ty::F32 => "float",
                Ty::F16 => "half",
                _ => unimplemented!(),
            };
            let src = CodeGen::new(include_str!("swiglu.cl"))
                .define("Tval", dt)
                .to_string();
            KernelCache::new(&self.ctx, &src, CL2_0)
        });
        (key, group_size)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt: DigitLayout,
    d: usize,
}

#[cfg(test)]
mod test {
    use super::{Args, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F32, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::{null, null_mut};
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,
        gate_base: *mut H::Byte,
        up_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt, &[n, d]);
        Args {
            gate_layout: layout.clone(),
            gate_base,
            up_layout: layout,
            up_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            opencl::ClDevice,
            test_utils::{Diff, ErrorCollector},
        };
        use clrt::Platform;
        use rand::Rng;
        use std::{iter::zip, time::Instant};

        let mut cpu_op = RefOp::new(&Cpu);
        for platform in Platform::all() {
            for device in platform.devices() {
                println!("device: {}", device.name());

                let context = device.context();
                let queue = context.queue();
                let mut cl_op = Operator::new(&ClDevice::new(context.clone(), Default::default()));
                cpu_op.scheme(&dyn_args(F64), 0).unwrap();
                cl_op.scheme(&dyn_args(F32), 0).unwrap();

                // let n = 5632;
                // let d = 2048;
                let n = 1;
                let d = 5632;
                let mut gate = vec![0.0f64; n * d];
                let mut up = vec![0.0f64; n * d];
                rand::rng().fill(&mut gate[..]);
                rand::rng().fill(&mut up[..]);
                let up = up;

                let mut gate_svm = context.malloc::<f32>(n * d);
                let mut up_svm = context.malloc::<f32>(n * d);

                let mut map = queue.map_mut(&mut gate_svm, false);
                let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                    panic!()
                };
                for (dst, src) in zip(mem, &gate) {
                    *dst = *src as _;
                }
                queue.unmap(map);
                let mut map = queue.map_mut(&mut up_svm, false);
                let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                    panic!()
                };
                for (dst, src) in zip(mem, &up) {
                    *dst = *src as _;
                }
                queue.unmap(map);

                let time = Instant::now();
                cl_op
                    .launch(
                        &args(
                            F32,
                            n,
                            d,
                            gate_svm.as_mut_ptr().cast(),
                            up_svm.as_ptr().cast(),
                        ),
                        &mut [],
                        &queue,
                    )
                    .unwrap();
                queue.finish();
                let cl_time = time.elapsed();

                let time = Instant::now();
                cpu_op
                    .launch(
                        &args(F64, n, d, gate.as_mut_ptr().cast(), up.as_ptr().cast()),
                        &mut [],
                        &ThisThread,
                    )
                    .unwrap();
                let cpu_time = time.elapsed();
                let map = queue.map(&mut gate_svm);
                let ([], y_ans, []) = (unsafe { map.align_to::<f32>() }) else {
                    panic!()
                };
                let diff = gate
                    .into_iter()
                    .zip(y_ans)
                    .map(|(a, b)| Diff::new(a, *b as _))
                    .collect::<Vec<_>>();
                queue.unmap(map);

                let mut ec = ErrorCollector::new(f32::EPSILON as f64, 0.);
                diff.into_iter().for_each(|diff| ec.push(diff));
                println!("{ec}");
                println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
                let (out, count) = ec.summary();
                assert!(out * 1000 <= count);
            }
        }
    }
}
