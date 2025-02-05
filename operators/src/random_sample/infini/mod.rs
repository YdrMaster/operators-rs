use super::{args::Meta, common_cpu::Operator as RefOp, Args, Indices, RandomSample};
use crate::{
    common_cpu::{Cpu, ThisThread},
    get_static,
    infini::Device,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use std::{ptr::null, slice::from_raw_parts};

pub struct Operator(Device);

impl RandomSample<Device> for Operator {
    fn build_indices<QA>(_n: usize, queue_alloc: &QA) -> Indices<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        Indices {
            n: 0,
            mem: queue_alloc.alloc(1),
        }
    }
}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    fn new(node: &Self::TopoNode) -> Self {
        Self(node.clone())
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
        let Args {
            kv_pair: _,
            kv_pair_base,
            logits: _,
            logits_base,
            indices: _,
            indices_base: _,
            config,
            seed,
        } = args;
        let Meta { dt, n } = args.meta()?;
        get_static! {
            n
        }
        let unit = dt.nbytes();
        let mut host = vec![0u8; n * unit];

        self.0
            .memcpy_d2h(&mut host, unsafe { from_raw_parts(*logits_base, n * unit) });

        let cpu_op = RefOp::new(&Cpu);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: kv_pair_base.cast(),
                    logits_base: host.as_ptr().cast(),
                    indices_base: null(),
                    config: *config,
                    seed: *seed,
                    ..Args::layout(dt, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
        Ok(())
    }
}

#[test]
fn test_compute() {
    use super::args::SampleArgs;
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        infini::cast_load,
        Operator as _,
    };
    use digit_layout::types as ty;
    use rand::Rng;
    use std::ptr::null;
    infini_rt::init(infini_rt::DEVICE_CPU);
    let dev = Device::cpu();

    let cpu_op = RefOp::new(&Cpu);
    let mut dev_op = Operator::new(&dev);
    let n = 32000;

    dev_op
        .scheme(&Args::layout(ty::F32, n), usize::MAX)
        .unwrap();

    let mut logits = vec![0.0f32; n];
    rand::rng().fill(&mut logits[..]);

    // argmax
    {
        let kv_ans = {
            let stream = dev.stream();
            let logits = cast_load(&logits, |x| x as f32, &stream);
            let mut kv: KVPair<f32> = KVPair::new(u32::MAX, 0.0f32);

            dev_op
                .launch(
                    &Args {
                        kv_pair_base: (&mut kv) as *mut _ as _,
                        logits_base: logits.as_ptr().cast(),
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &stream,
                )
                .unwrap();
            kv
        };

        let mut kv_ref: KVPair<f32> = KVPair::new(u32::MAX, 0.0f32);
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
        let kv_ans = {
            let stream = dev.stream();

            let logits = stream.from_host(&logits);
            let indices = Operator::build_indices(n, &stream).mem;
            let mut kv = KVPair::new(0, 0.0f32);

            dev_op
                .launch(
                    &Args {
                        kv_pair_base: (&mut kv) as *mut _ as _,
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
            kv
        };

        let mut kv_ref = KVPair::new(0, 0.0f32);
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
