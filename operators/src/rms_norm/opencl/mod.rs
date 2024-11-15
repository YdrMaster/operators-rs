use super::{args::Meta, Args, RmsNorm};
use crate::{
    execution_failed, get_static, opencl::ClDevice, shape_not_support, ByteOf, LaunchError,
    QueueAlloc, SchemeError,
};
use clrt::Kernel;
use std::{collections::HashMap, ffi::CString};

pub struct Operator {
    kernels: HashMap<String, Kernel>,
    kernel_name: Option<String>,
}

impl RmsNorm<ClDevice> for Operator {}

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(_node: &Self::TopoNode) -> Self {
        let options = CString::new("").unwrap();
        let program = _node
            .context()
            .build_from_source(include_str!("rms_norm.cl"), options);
        let kernel_names = [
            "rms_norm_padding",
            "rms_norm_folding",
            "rms_norm_padding_f16",
            "rms_norm_folding_f16",
        ];
        let mut kernels = HashMap::new();

        for name in kernel_names {
            let c_name = CString::new(name).expect("CString creation failed");
            if let Some(kernel) = program.get_kernel(&c_name) {
                kernels.insert(name.to_string(), kernel);
            } else {
                eprintln!("Kernel '{}' not found", name);
            }
        }
        Self {
            kernels,
            kernel_name: None,
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { d, .. } = _args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            w_layout: _,
            w_base,
            epsilon,
        } = _args;
        let &[nsy, ..] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, ..] = x_layout.strides() else {
            unreachable!()
        };

        get_static! {
            d
            nsy nsx
        }

        let max_threads_per_block = 512;
        let items_per_thread = (d + max_threads_per_block - 1) / max_threads_per_block;
        let kernel_name = match items_per_thread {
            1 => "rms_norm_padding",
            2..=16 => "rms_norm_folding",
            // 1 => "rms_norm_padding_f16",
            // 2..=16 => "rms_norm_folding_f16",
            _ => {
                // todo!() 添加倍数大于16的处理
                return Err(shape_not_support("Unsupported items_per_thread configuration").into());
            }
        };

        let kernel = self
            .kernels
            .get_mut(kernel_name)
            .ok_or_else(|| execution_failed("Kernel not found"))
            .unwrap();
        use clrt::bindings::cl_int;
        kernel
            .set_arg(0, y_base)
            .set_arg(1, (nsy / 4) as cl_int)
            .set_arg(2, x_base)
            .set_arg(3, (nsx / 4) as cl_int)
            .set_arg(4, w_base)
            .set_arg(5, epsilon);
        println!("x is of type f32");
        self.kernel_name = Some(kernel_name.to_string());
        Ok(0)
    }

    fn launch<QA>(
        &self,
        _args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { n, d, .. } = _args.meta()?;
        get_static! {
            n   d
        }
        let max_threads_per_block = 512;
        let items_per_thread = (d + max_threads_per_block - 1) / max_threads_per_block;
        let local_worksize_y = match items_per_thread {
            1 => d as usize,
            2..=16 => max_threads_per_block as usize,
            _ => {
                // todo!() 添加倍数大于16的处理  --worksize存入operator中，在scheme处理
                return Err(shape_not_support("Unsupported items_per_thread configuration").into());
            }
        };
        let global_workoffset = [0];
        let global_worksize = [(n * d) as usize];
        let local_worksize = [local_worksize_y];

        self.kernels
            .get(self.kernel_name.as_ref().unwrap())
            .unwrap()
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                _queue_alloc.queue(),
                None,
            );
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, ClDevice, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use clrt::Platform;
    use digit_layout::{
        types::{F32, F64},
        DigitLayout,
    };
    use rand::Rng;

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
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            test_utils::{Diff, ErrorCollector},
        };
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let platform = Platform::all()
            .into_iter()
            .next()
            .expect("No platform found");
        let devices = platform.devices();
        let context = devices[0].context();
        let queue = context.queue();
        let cl_device = ClDevice::new(context.clone());
        let mut gpu_op = Operator::new(&cl_device);
        let mut cpu_op = RefOp::new(&Cpu);

        for k in 8..=8 {
            let n = 4;
            let d = 1 << k;

            cpu_op.scheme(&dyn_args(F64, F64, d), 0).unwrap();

            let mut y = vec![0.0f32; n * d];
            let mut x = vec![0.0f64; n * d];
            let mut w = vec![0.0f64; d];

            rand::thread_rng().fill(&mut x[..]);
            rand::thread_rng().fill(&mut w[..]);

            let mut y_svm = context.malloc::<f32>(n * d);
            let mut x_svm = context.malloc::<f32>(n * d);
            let mut w_svm = context.malloc::<f32>(d);
            let x_f32: Vec<f32> = x.iter().map(|&val| val as f32).collect();
            let w_f32: Vec<f32> = w.iter().map(|&val| val as f32).collect();

            // queue.memcpy_from_host(&mut x_svm, &x_f32, None);
            // queue.memcpy_from_host(&mut w_svm, &w_f32, None);
            // queue.finish();
            use clrt::Valid;
            use std::slice::from_raw_parts_mut;
            let mut map = queue.map_mut(&mut x_svm, Valid);
            {
                let mem = unsafe {
                    from_raw_parts_mut(map.as_mut_ptr().cast::<f32>(), map.len() / size_of::<f32>())
                };
                mem.copy_from_slice(&x_f32);
            }
            queue.unmap(map);
            let mut map = queue.map_mut(&mut w_svm, Valid);
            {
                let mem = unsafe {
                    from_raw_parts_mut(map.as_mut_ptr().cast::<f32>(), map.len() / size_of::<f32>())
                };
                mem.copy_from_slice(&w_f32);
            }
            queue.unmap(map);

            gpu_op
                .scheme(
                    &args(
                        F32,
                        F32,
                        n,
                        d,
                        y_svm.as_mut_ptr().cast(),
                        x_svm.as_ptr().cast(),
                        w_svm.as_ptr().cast(),
                    ),
                    0,
                )
                .unwrap();
            gpu_op
                .launch(
                    &args(
                        F32,
                        F32,
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
            let mut map = queue.map_mut(&mut y_svm, Valid);
            {
                let mem = unsafe {
                    from_raw_parts_mut(map.as_mut_ptr().cast::<f32>(), map.len() / size_of::<f32>())
                };
                y.copy_from_slice(&mem);
            }
            queue.unmap(map);
            // queue.memcpy_to_host(&mut y, &y_svm, None);
            // queue.finish();
            println!("y 数组：");
            for i in &y {
                print!("{} ", i);
            }
            println!();
            //CPU
            let mut y_ref = vec![0.0; n * d];
            cpu_op
                .launch(
                    &args(
                        F64,
                        F64,
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
            for i in &y_ref {
                print!("{} ", i);
            }
            println!();
            let diff = y_ref
                .into_par_iter()
                .zip(y)
                .map(|(a, b)| Diff::new(a, b as f64))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f32::EPSILON as f64, 1e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
