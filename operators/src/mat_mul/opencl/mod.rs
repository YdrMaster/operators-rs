use super::{args::SchemeLayout, Args, MatMul};
use crate::{
    opencl::{ClDevice, KernelCache, CL2_0},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_int;

#[repr(transparent)]
pub struct Operator(KernelCache);

impl MatMul<ClDevice> for Operator {}

const MAX_THREADS_PER_BLOCK: usize = 512;

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(node: &Self::TopoNode) -> Self {
        Self(KernelCache::new(
            node.context(),
            include_str!("mat_mul.cl"),
            CL2_0,
        ))
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
        let SchemeLayout {
            ab_swap,
            a_trans,
            b_trans,
            batch,
            m,
            n,
            k,
            c_stride,
            c_ld,
            a_stride,
            a_ld,
            b_stride,
            b_ld,
            ..
        } = args.layout()?;
        let &Args {
            c_base,
            beta,
            a_base,
            b_base,
            alpha,
            ..
        } = args;

        let [a, b] = if ab_swap {
            [b_base, a_base]
        } else {
            [a_base, b_base]
        };
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        let mn = m * n;
        let items_per_thread = mn.div_ceil(MAX_THREADS_PER_BLOCK);
        let local_worksize_y = match items_per_thread {
            1 => mn,
            _ => MAX_THREADS_PER_BLOCK,
        };
        let name = "general_gemm_f32";

        let mut kernel = self.0.take(name).unwrap();
        let queue = _queue_alloc.queue();
        kernel
            .set_arg(0, a)
            .set_arg(1, b)
            .set_arg(2, c_base)
            .set_arg(3, a_stride as cl_int)
            .set_arg(4, lhs_rs as cl_int)
            .set_arg(5, lhs_cs as cl_int)
            .set_arg(6, b_stride as cl_int)
            .set_arg(7, rhs_rs as cl_int)
            .set_arg(8, rhs_cs as cl_int)
            .set_arg(9, c_stride as cl_int)
            .set_arg(10, 1 as cl_int)
            .set_arg(11, c_ld as cl_int)
            .set_arg(12, batch as cl_int)
            .set_arg(13, m as cl_int)
            .set_arg(14, n as cl_int)
            .set_arg(15, k as cl_int)
            .set_arg(16, alpha)
            .set_arg(17, beta) //;
            .launch(&[0, 0], &[batch, mn], &[1, local_worksize_y], queue, None);
        self.0.put(name, kernel);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    const ALPHA: f32 = 0.5;
    const BETA: f32 = 1.;

    fn args<H: Hardware>(
        dt: DigitLayout,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[batch, m, n]),
            c_base,
            beta: BETA,
            a_layout: TensorLayout::new_contiguous(dt, &[batch, m, k]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[batch, k, n]),
            b_base,
            alpha: ALPHA,
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
        use digit_layout::types::{F32, F64};
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
        use std::{iter::zip, time::Instant};

        let cpu_op = RefOp::new(&Cpu);
        for platform in Platform::all() {
            for device in platform.devices() {
                println!("device: {}", device.name());

                let context = device.context();
                let queue = context.queue();
                let cl_op = Operator::new(&ClDevice::new(context.clone(), Default::default()));

                let batch = 4;
                let k = 9;
                let n = 64;
                for m in [8] {
                    let mut a = vec![0.0f64; batch * m * k];
                    let mut b = vec![0.0f64; batch * k * n];
                    let mut c = vec![0.0f64; batch * m * n];

                    rand::rng().fill(&mut a[..]);
                    rand::rng().fill(&mut b[..]);
                    rand::rng().fill(&mut c[..]);

                    let a = a;
                    let b = b;
                    let mut a_svm = context.malloc::<f32>(batch * m * k);
                    let mut b_svm = context.malloc::<f32>(batch * k * n);

                    let mut map = queue.map_mut(&mut a_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(mem, &a) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let mut map = queue.map_mut(&mut b_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(mem, &b) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let mut c_svm = context.malloc::<f32>(batch * m * n);
                    let mut map = queue.map_mut(&mut c_svm, false);
                    let ([], mem, []) = (unsafe { map.align_to_mut::<f32>() }) else {
                        panic!()
                    };
                    for (dst, src) in zip(mem, &c) {
                        *dst = *src as _;
                    }
                    queue.unmap(map);

                    let time = Instant::now();
                    cl_op
                        .launch(
                            &args(
                                F32,
                                batch,
                                m,
                                n,
                                k,
                                c_svm.as_mut_ptr().cast(),
                                a_svm.as_ptr().cast(),
                                b_svm.as_ptr().cast(),
                            ),
                            &mut [],
                            &queue,
                        )
                        .unwrap();
                    queue.finish();

                    let cl_time = time.elapsed();

                    let time = Instant::now();
                    let mut c_ref = c;
                    cpu_op
                        .launch(
                            &args(
                                F64,
                                batch,
                                m,
                                n,
                                k,
                                c_ref.as_mut_ptr().cast(),
                                a.as_ptr().cast(),
                                b.as_ptr().cast(),
                            ),
                            &mut [],
                            &ThisThread,
                        )
                        .unwrap();
                    let cpu_time = time.elapsed();

                    let map = queue.map(&mut c_svm);
                    let ([], y_ans, []) = (unsafe { map.align_to::<f32>() }) else {
                        panic!()
                    };
                    let diff = c_ref
                        .into_par_iter()
                        .zip(y_ans)
                        .map(|(a, b)| Diff::new(a, *b as _))
                        .collect::<Vec<_>>();
                    let mut ec = ErrorCollector::new(f32::EPSILON as f64, 5e-3);
                    diff.into_iter().for_each(|diff| ec.push(diff));
                    println!("{ec}");
                    println!("cl: {cl_time:?} / cpu: {cpu_time:?}");

                    let (out, count) = ec.summary();
                    queue.unmap(map);
                    assert!(out * 1000 <= count);
                }
            }
        }
    }
}
