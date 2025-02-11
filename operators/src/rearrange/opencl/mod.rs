use super::{args::Scheme, Args, Rearrange};
use crate::{
    opencl::{ClDevice, CodeGen, KernelCache, CL2_0},
    rank_not_support, ByteOf, LaunchError, QueueAlloc,
    SchemeDiversity::Low as LowDiversity,
    SchemeError,
};
use clrt::{bindings::cl_int, Context};
use lru::LruCache;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Mutex;

pub struct Operator {
    ctx: Context,
    max_group_size: usize,
    schemes: Mutex<LruCache<SchemeKey, KernelCache>>,
}

impl Rearrange<ClDevice> for Operator {}

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
        let scheme = Scheme::new(args)?;
        let unit = scheme.unit();
        if scheme.count() == 1 {
            let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let src = unsafe { from_raw_parts(args.src_base, unit) };
            queue_alloc.queue().memcpy(dst, src, None);
            return Ok(());
        }

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
            _ => Err(rank_not_support(
                "rearrange not support ndim > 2 on Mobile GPU",
            ))?,
        };
        let unit_size = unit / 4;
        let (key, group_size) = self.cache_kernel(unit_size);
        let mut rearrange = self
            .schemes
            .lock()
            .unwrap()
            .get(&key)
            .unwrap()
            .take("rearrange")
            .unwrap();

        let unit = unit as i32;
        let dst_rs = dst_rs / unit;
        let dst_cs = dst_cs / unit;
        let src_rs = src_rs / unit;
        let src_cs = src_cs / unit;

        rearrange
            .set_arg(0, args.dst_base)
            .set_arg(1, dst_rs as cl_int)
            .set_arg(2, dst_cs as cl_int)
            .set_arg(3, args.src_base)
            .set_arg(4, src_rs as cl_int)
            .set_arg(5, src_cs as cl_int)
            .set_arg(6, c as cl_int)
            .set_arg(7, unit_size as cl_int)
            .launch(
                &[0],
                &[(r * c * (unit_size as u32)) as usize],
                &[group_size],
                queue_alloc.queue(),
                None,
            );

        let mut cache = self.schemes.lock().unwrap();
        let program = cache.get(&key).unwrap();
        program.put("rearrange", rearrange);
        Ok(())
    }
}

impl Operator {
    fn cache_kernel(&self, unit_size: usize) -> (SchemeKey, usize) {
        let items_per_thread = unit_size.div_ceil(self.max_group_size);
        let group_size = match items_per_thread {
            1 => unit_size,
            _ => self.max_group_size,
        };
        let key = SchemeKey { unit_size };
        self.schemes.lock().unwrap().get_or_insert(key, || {
            let src = CodeGen::new(include_str!("rearrange.cl")).to_string();
            KernelCache::new(&self.ctx, &src, CL2_0)
        });
        (key, group_size)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    unit_size: usize,
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
        use clrt::Platform;
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
                let mut cl_op = Operator::new(&ClDevice::new(context.clone(), Default::default()));

                let nh = 5;
                let seq = 32;
                let dh = 64;
                let mut src = vec![0u32; nh * seq * dh];
                rand::rng().fill(&mut src[..]);
                let s_src =
                    ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, dt.nbytes());
                let s_dst =
                    ArrayLayout::<3>::new_contiguous(&[seq, nh, dh], BigEndian, dt.nbytes())
                        .transpose(&[1, 0]);

                let dt = ty::U32;
                cpu_op.scheme(&dyn_args(dt), 0).unwrap();
                cl_op.scheme(&dyn_args(dt), 0).unwrap();

                let mut s_svm = context.malloc::<u32>(nh * seq * dh * 2);
                let mut d_svm = context.malloc::<u32>(nh * seq * dh);

                let mut map = queue.map_mut(&mut s_svm, false);
                let ([], mem, []) = (unsafe { map.align_to_mut::<u32>() }) else {
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
                assert_eq!(y_ans, dst_ref);
                queue.unmap(map);
                println!("cl: {cl_time:?} / cpu: {cpu_time:?}");
            }
        }
    }
}
