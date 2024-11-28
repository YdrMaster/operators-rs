﻿use super::{args::Meta, Args, FusedSoftmax};
use crate::{
    get_static,
    opencl::{ClDevice, KernelCache},
    shape_not_support, type_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_int;
use digit_layout::types::F32;
use std::ffi::CString;

pub struct Operator(KernelCache);

impl FusedSoftmax<ClDevice> for Operator {}

const MAX_THREADS_PER_BLOCK: usize = 512;

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(_node: &Self::TopoNode) -> Self {
        let options = CString::new("").unwrap();
        let program = _node
            .context()
            .build_from_source(include_str!("fuesd_softmax.cl"), options);
        Self(KernelCache::new(program))
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt } = _args.meta()?;
        let Args { att_layout, .. } = _args;

        if dt != F32 {
            return Err(type_not_support("").into());
        }

        let &[att_len, ..] = att_layout.shape() else {
            unreachable!()
        };
        let Some(att_len) = att_len.get_static() else {
            return Ok(0);
        };

        let items_per_thread = att_len.div_ceil(MAX_THREADS_PER_BLOCK);
        let kernel_name = match items_per_thread {
            1 => "softmax_padding",
            2..=16 => "softmax_folding",
            // todo!() 添加倍数大于16的处理
            _ => {
                return Err(shape_not_support(
                    "Unsupported items_per_thread configuration",
                ))
            }
        };

        self.0
            .set_kernel(kernel_name, self.0.get_kernel(kernel_name).unwrap());
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
        let Meta { dt } = _args.meta()?;
        let Args {
            att_layout,
            att_base,
        } = _args;
        let &[nh, seq_len, att_len] = att_layout.shape() else {
            unreachable!()
        };
        if dt != F32 {
            return Err(type_not_support("").into());
        }
        get_static! {
            nh seq_len att_len
        }

        let items_per_thread = att_len.div_ceil(MAX_THREADS_PER_BLOCK);
        let (name, local_worksize_y) = match items_per_thread {
            1 => ("softmax_padding", att_len),
            2..=16 => ("softmax_folding", MAX_THREADS_PER_BLOCK),
            _ => {
                // todo!() 添加倍数大于16的处理
                return Err(shape_not_support("Unsupported items_per_thread configuration").into());
            }
        };

        let global_workoffset = [0];
        let global_worksize = [(nh * seq_len * local_worksize_y) as usize];
        let local_worksize = [local_worksize_y];

        let mut kernel = self.0.get_kernel(name).unwrap();

        kernel
            .set_arg(0, att_base)
            .set_arg(1, seq_len as cl_int)
            .set_arg(2, att_len as cl_int)
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                _queue_alloc.queue(),
                None,
            );

        self.0.set_kernel(name, kernel);
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
        use clrt::{Invalid, Platform};
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
                cpu_op.scheme(&dyn_args(ty::F64), 0).unwrap();
                cl_op.scheme(&dyn_args(ty::F32), 0).unwrap();

                let nh = 3;
                for (seq_len, att_len) in [(7, 511)] {
                    let mut att = vec![0.0f64; nh * seq_len * att_len];
                    rand::thread_rng().fill(&mut att[..]);
                    let mut att_svm = context.malloc::<f32>(nh * seq_len * att_len);
                    let mut map = queue.map_mut(&mut att_svm, Invalid);
                    let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<f32>() })
                    else {
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
                    for (index, i) in mem.iter().enumerate() {
                        print!("{}: {} ", index + 1, i);
                    }
                    println!();
                    println!();
                    for (index, i) in att.iter().enumerate() {
                        print!("{}: {} ", index + 1, i);
                    }
                    println!();
                    println!();
                    let diff = att
                        .into_par_iter()
                        .zip(mem)
                        .map(|(a, b)| Diff::new(a, *b as _))
                        .collect::<Vec<_>>();
                    queue.unmap(map);

                    let mut ec = ErrorCollector::new(f32::EPSILON as f64, 1e-3);
                    diff.into_iter().for_each(|diff| ec.push(diff));
                    println!("{ec}");
                    println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
                    // let ee = ec.outliers();
                    // println!("ee: {ee:?}");

                    let (out, count) = ec.summary();
                    assert!(out * 1000 <= count);
                    // assert!(1 <= 2); //测试性能
                }
            }
        }
    }
}
