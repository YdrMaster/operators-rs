use super::{args::Meta, Args, FusedSoftmax};
use crate::{
    get_static,
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    type_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_uint;
use digit_layout::types as Ty;

pub struct Operator {
    group_size: usize,
    kernel: KernelCache,
}

impl FusedSoftmax<ClDevice> for Operator {}

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(node: &Self::TopoNode) -> Self {
        let ctx = node.context();
        let group_size = ctx
            .devices()
            .iter()
            .map(|d| d.max_group_size())
            .min()
            .unwrap()
            / 2; // 直接用最大 group 可能导致资源不足
        let src = CodeGen::new(include_str!("fuesd_softmax.cl"))
            .define("GROUP_SIZE", group_size)
            .to_string();
        Self {
            group_size,
            kernel: KernelCache::new(ctx, &src, CL2_0),
        }
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
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt } = args.meta()?;
        let Args {
            att_layout,
            att_base,
        } = args;
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };

        if dt != Ty::F32 {
            return Err(type_not_support("").into());
        }
        get_static! {
            nh seq_len att_len
        }

        let items_per_thread = att_len.div_ceil(self.group_size);
        let (name, local_worksize_y) = match items_per_thread {
            1 => ("softmax_padding", att_len),
            2..=16 => ("softmax_folding", self.group_size),
            _ => ("softmax_general", self.group_size),
        };
        let localsize = local_worksize_y.next_power_of_two(); //for padding block reduce

        let mut kernel = self.kernel.take(name).unwrap();

        kernel
            .set_arg(0, att_base)
            .set_arg(1, seq_len as cl_uint)
            .set_arg(2, att_len as cl_uint)
            .launch(
                &[0],
                &[nh * seq_len * localsize],
                &[localsize],
                queue_alloc.queue(),
                None,
            );

        self.kernel.put(name, kernel);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::null_mut;
        Args {
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
                    // for (seq_len, att_len) in [(1, 13)] {
                    // for (seq_len, att_len) in [(1, 1024), (1, 2048), (7, 2048)] {
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
                    println!("cl: {cl_time:?} / cpu: {cpu_time:?}");

                    let (out, count) = ec.summary();
                    assert!(out * 1000 <= count);
                }
            }
        }
    }
}
