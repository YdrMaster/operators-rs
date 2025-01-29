use super::{args::Meta, Args, RmsNorm};
use crate::{
    get_static,
    opencl::{ClDevice, KernelCache, CL2_0},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_int;

#[repr(transparent)]
pub struct Operator(KernelCache);

impl RmsNorm<ClDevice> for Operator {}

const MAX_THREADS_PER_BLOCK: usize = 512;

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(node: &Self::TopoNode) -> Self {
        Self(KernelCache::new(
            node.context(),
            include_str!("rms_norm.cl"),
            CL2_0,
        ))
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { d, .. } = args.meta()?;
        let Some(d) = d.get_static() else {
            return Ok(0);
        };

        let items_per_thread = d.div_ceil(MAX_THREADS_PER_BLOCK);
        let kernel_name = match items_per_thread {
            1 => "rms_norm_padding",
            2..=16 => "rms_norm_folding",
            _ => "rms_norm_general",
        };

        self.0
            .set_kernel(kernel_name, self.0.get_kernel(kernel_name).unwrap());
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
        let Meta { n, d, .. } = args.meta()?;
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
            n nsy nsx d
        }

        let items_per_thread = d.div_ceil(MAX_THREADS_PER_BLOCK);
        let (name, local_worksize_y) = match items_per_thread {
            1 => ("rms_norm_padding", d),
            2..=16 => ("rms_norm_folding", MAX_THREADS_PER_BLOCK),
            _ => ("rms_norm_general", MAX_THREADS_PER_BLOCK),
        };
        let global_workoffset = [0];
        let global_worksize = [(n * local_worksize_y) as usize];
        let local_worksize = [local_worksize_y];

        let mut kernel = self.0.get_kernel(name).unwrap();
        _queue_alloc.queue().finish();

        kernel
            .set_arg(0, y_base)
            .set_arg(1, (nsy / 4) as cl_int)
            .set_arg(2, x_base)
            .set_arg(3, (nsx / 4) as cl_int)
            .set_arg(4, w_base)
            .set_arg(5, epsilon);
        if name == "rms_norm_folding" || name == "rms_norm_general" {
            kernel.set_arg(6, d as cl_int);
        }
        kernel.launch(
            &global_workoffset,
            &global_worksize,
            &local_worksize,
            _queue_alloc.queue(),
            None,
        );
        _queue_alloc.queue().finish();

        self.0.set_kernel(name, kernel);
        Ok(())
    }
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
                let mut cl_op = Operator::new(&ClDevice::new(context.clone()));

                for k in 2..=12 {
                    let n = 5;
                    let d = 1 << k;

                    cpu_op.scheme(&dyn_args(ty::F64, ty::F64, d), 0).unwrap();
                    cl_op.scheme(&dyn_args(ty::F32, ty::F32, d), 0).unwrap();

                    let mut x = vec![0.0f64; n * d];
                    let mut w = vec![0.0f64; d];
                    rand::thread_rng().fill(&mut x[..]);
                    rand::thread_rng().fill(&mut w[..]);

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
