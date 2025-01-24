use super::{args::Meta, Args, Indices, RandomSample};
use crate::{
    get_static,
    opencl::{ClDevice, KernelCache},
    strides_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};

use clrt::bindings::cl_int;
use std::ffi::CString;

pub struct Operator(KernelCache);

const MAX_THREADS_PER_BLOCK: usize = 256;

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
        const SRC: &str = include_str!("random_sample.cl");
        let opts = CString::new("").unwrap();
        Self(KernelCache::new(node.context(), SRC, &opts))
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt, n } = args.meta()?;
        let &[s] = args.logits.strides() else {
            unreachable!()
        };
        let Args {
            kv_pair_base,
            logits_base,
            ..
        } = args;
        let unit = dt.nbytes() as isize;
        if s.get_static().copied() != Some(unit) {
            return Err(strides_not_support("").into());
        }

        get_static!(n);

        let name = "argmax_step1";
        let global_workoffset = [0];
        let global_worksize = [n as usize];
        let local_worksize = [MAX_THREADS_PER_BLOCK];

        let mut kernel = self.0.get_kernel(name).unwrap();
        let queue = _queue_alloc.queue();

        kernel
            .set_arg(0, logits_base)
            .set_arg(1, n as cl_int)
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                queue,
                None,
            );

        self.0.set_kernel(name, kernel);
        let name = "argmax_step2";
        let len = n / MAX_THREADS_PER_BLOCK;
        let global_workoffset = [0];
        let global_worksize = [256 as usize];
        let local_worksize = [256];
        let mut kernel = self.0.get_kernel(name).unwrap();

        kernel
            .set_arg(0, logits_base)
            .set_arg(1, kv_pair_base)
            .set_arg(2, len as cl_int)
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                queue,
                None,
            );
        self.0.set_kernel(name, kernel);

        Ok(())
    }
}

#[test]
fn test_compute() {
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        opencl::ClDevice,
        Operator as _,
    };
    use clrt::{Invalid, Platform};
    use digit_layout::types as ty;
    use rand::Rng;
    use std::{iter::zip, time::Instant};

    let n = 32000;

    let cpu_op = RefOp::new(&Cpu);
    for platform in Platform::all() {
        for device in platform.devices() {
            println!("device: {}", device.name());
            let context = device.context();
            let queue = context.queue();
            let mut cl_op = Operator::new(&ClDevice::new(context.clone()));

            let mut logits = vec![0.0f32; n];
            rand::thread_rng().fill(&mut logits[..]);
            let mut logits_svm = context.malloc::<f32>(n);
            let mut kv_pair_svm = context.malloc::<KVPair>(1);

            let mut map = queue.map_mut(&mut logits_svm, Invalid);
            let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<f32>() }) else {
                panic!()
            };
            for (dst, src) in zip(mem, &logits) {
                *dst = *src;
            }
            queue.unmap(map);
            std::thread::sleep(std::time::Duration::from_secs(1));

            for (index, i) in logits.iter().enumerate() {
                print!("{}: {} ", index + 1, i);
            }
            println!();
            println!();

            let map = queue.map(&mut logits_svm);
            let ([], y_ans, []) = (unsafe { map.align_to::<f32>() }) else {
                panic!()
            };
            for (index, i) in y_ans.iter().enumerate() {
                print!("{}: {} ", index + 1, i);
            }
            println!();
            println!();
            queue.unmap(map);

            let time = Instant::now();
            cl_op
                .launch(
                    &Args {
                        kv_pair_base: kv_pair_svm.as_mut_ptr().cast(),
                        logits_base: logits_svm.as_ptr().cast(),
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &queue,
                )
                .unwrap();
            queue.finish();
            let cl_time = time.elapsed();

            let mut kv_ref = KVPair::new(u32::MAX, f32::MAX);
            let time = Instant::now();
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
            let cpu_time = time.elapsed();

            println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
            let mut map = queue.map(&mut kv_pair_svm);

            let ([], y_ans, []) = (unsafe { map.align_to::<KVPair>() }) else {
                panic!()
            };
            assert_eq!(y_ans[0].idx(), kv_ref.idx());
            queue.unmap(map);
        }
    }
}
