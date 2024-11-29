use super::{args::Scheme, Args, Rearrange};
use crate::{
    opencl::{ClDevice, KernelCache},
    rank_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use clrt::bindings::cl_int;
use std::{
    ffi::CString,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Operator(KernelCache);

impl Rearrange<ClDevice> for Operator {}

impl crate::Operator for Operator {
    type Hardware = ClDevice;
    type TopoNode = ClDevice;
    type Args = Args<ClDevice>;

    fn new(node: &Self::TopoNode) -> Self {
        // let options = CString::new("").unwrap();
        // let program = _node
        //     .context()
        //     .build_from_source(include_str!("rearrange.cl"), options);
        // Self(KernelCache::new(program))
        const SRC: &str = include_str!("rearrange.cl");
        let opts = CString::new("").unwrap();
        Self(KernelCache::new(node.context(), SRC, &opts))
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let kernel_name = "rearrange";
        self.0
            .set_kernel(kernel_name, self.0.get_kernel(kernel_name).unwrap());
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
        let scheme = Scheme::new(args)?;
        if scheme.ndim() == 0 {
            let unit = scheme.unit();
            let _dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let _src = unsafe { from_raw_parts(args.src_base, unit) };
            // queue_alloc.queue().memcpy_d2d(dst, src);
            //todo queue添加d2d接口函数
            return Ok(());
        }
        let scheme = scheme.distribute_unit((0..=5).rev().map(|n| 32 * (1 << n)));
        let unit = scheme.unit();

        struct Layout {
            r: u32,
            c: u32,
            dst_rs: i32,
            dst_cs: i32,
            src_rs: i32,
            src_cs: i32,
        }

        let Layout {
            r,
            c,
            dst_rs,
            dst_cs,
            src_rs,
            src_cs,
        } = match scheme.ndim() {
            0 => unreachable!(),
            1 => {
                let &[dst_cs] = scheme.dst_strides() else {
                    unreachable!()
                };
                let &[src_cs] = scheme.src_strides() else {
                    unreachable!()
                };
                Layout {
                    r: 1,
                    c: scheme.shape().next().unwrap() as _,
                    dst_rs: 0,
                    dst_cs: dst_cs as _,
                    src_rs: 0,
                    src_cs: src_cs as _,
                }
            }
            2 => {
                let mut shape = scheme.shape();
                let r = shape.next().unwrap();
                let c = shape.next().unwrap();
                let &[dst_rs, dst_cs] = scheme.dst_strides() else {
                    unreachable!()
                };
                let &[src_rs, src_cs] = scheme.src_strides() else {
                    unreachable!()
                };
                Layout {
                    r: r as _,
                    c: c as _,
                    dst_rs: dst_rs as _,
                    dst_cs: dst_cs as _,
                    src_rs: src_rs as _,
                    src_cs: src_cs as _,
                }
            }
            _ => Err(rank_not_support("rearrange not support ndim > 2 on NV GPU"))?,
        };

        let name = "rearrange";
        let global_workoffset = [0];
        let global_worksize = [(r * c * (unit as u32) / 128) as usize];
        let local_worksize = [(unit / 128) as usize]; //32*4字节

        let mut kernel = self.0.get_kernel(name).unwrap();

        let unit = unit as i32;
        let dst_rs = dst_rs / unit;
        let dst_cs = dst_cs / unit;
        let src_rs = src_rs / unit;
        let src_cs = src_cs / unit;
        let items = 32;

        kernel
            .set_arg(0, args.dst_base)
            .set_arg(1, dst_rs as cl_int)
            .set_arg(2, dst_cs as cl_int)
            .set_arg(3, args.src_base)
            .set_arg(4, src_rs as cl_int)
            .set_arg(5, src_cs as cl_int)
            .set_arg(6, c as cl_int)
            .set_arg(7, items as cl_int)
            .launch(
                &global_workoffset,
                &global_worksize,
                &local_worksize,
                queue_alloc.queue(),
                None,
            );

        self.0.set_kernel(name, kernel);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{ConstPtr, Hardware, MutPtr, TensorLayout};
    use digit_layout::DigitLayout;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            dst_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        shape: &[usize],
        s_src: &[isize],
        s_dst: &[isize],
        src_base: ConstPtr<H>,
        dst_base: MutPtr<H>,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new(dt, shape, s_dst),
            dst_base,
            src_layout: TensorLayout::new(dt, shape, s_src),
            src_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::{super::common_cpu::Operator as RefOp, Operator};
        use crate::{
            common_cpu::{Cpu, ThisThread},
            opencl::ClDevice,
            Operator as _,
        };
        use clrt::{Invalid, Platform};
        use digit_layout::types as ty;
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;
        use std::{iter::zip, time::Instant};

        let dt = ty::U32;

        let mut cpu_op = RefOp::new(&Cpu);
        for platform in Platform::all() {
            for device in platform.devices() {
                println!("device: {}", device.name());

                let context = device.context();
                let queue = context.queue();
                let mut cl_op = Operator::new(&ClDevice::new(context.clone()));

                let nh = 32;
                let seq = 7;
                let dh = 128;
                let mut src = vec![0u32; nh * seq * dh];
                rand::thread_rng().fill(&mut src[..]);
                let s_src = ArrayLayout::<3>::new_contiguous(
                    &[nh, seq, dh],
                    BigEndian,
                    // dt.nbytes().unwrap(),
                    dt.nbytes(),
                );
                let s_dst = ArrayLayout::<3>::new_contiguous(
                    &[seq, nh, dh],
                    BigEndian,
                    // dt.nbytes().unwrap(),
                    dt.nbytes(),
                )
                .transpose(&[1, 0]);
                let dt = ty::U32;
                cpu_op.scheme(&dyn_args(dt), 0).unwrap();
                cl_op.scheme(&dyn_args(dt), 0).unwrap();

                let mut s_svm = context.malloc::<u32>(nh * seq * dh);
                let mut d_svm = context.malloc::<u32>(nh * seq * dh);

                let mut map = queue.map_mut(&mut s_svm, Invalid);
                let ([], mem, []) = (unsafe { map.write_only_slice().align_to_mut::<u32>() })
                else {
                    panic!()
                };
                for (dst, src) in zip(mem, &src) {
                    *dst = *src as _;
                }
                queue.unmap(map);
                let time = Instant::now();
                cl_op
                    .launch(
                        &args(
                            ty::F32,
                            &[nh, seq, dh],
                            s_src.strides(),
                            s_dst.strides(),
                            s_svm.as_ptr().cast(),
                            d_svm.as_mut_ptr().cast(),
                        ),
                        &mut [],
                        &queue,
                    )
                    .unwrap();
                queue.finish();
                let cl_time = time.elapsed();

                let mut dst_ref = vec![0u32; seq * nh * dh];
                let time = Instant::now();
                cpu_op
                    .launch(
                        &args(
                            dt,
                            &[nh, seq, dh],
                            s_src.strides(),
                            s_dst.strides(),
                            src.as_ptr().cast(),
                            dst_ref.as_mut_ptr().cast(),
                        ),
                        &mut [],
                        &ThisThread,
                    )
                    .unwrap();
                let cpu_time = time.elapsed();

                let map = queue.map(&mut d_svm);
                let ([], y_ans, []) = (unsafe { map.align_to::<u32>() }) else {
                    panic!()
                };
                // for (index, i) in y_ans.iter().enumerate() {
                //     print!("{}: {} ", index + 1, i);
                // }
                // println!();

                // for (index, i) in dst_ref.iter().enumerate() {
                //     print!("{}: {} ", index + 1, i);
                // }
                // println!();
                assert_eq!(y_ans, dst_ref);
                queue.unmap(map);
                println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
                // assert!(2 <= 1);
            }
        }
    }
}
