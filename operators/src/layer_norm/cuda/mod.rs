use super::{Args, LayerNorm};
use crate::{
    cuda::{dt_name, Gpu, Handle, ModuleBox},
    get_static,
    layer_norm::args::Meta,
    shape_not_support, strides_not_support, ByteOf, LaunchError, QueueAlloc, SchemeDiversity,
    SchemeError,
};
use digit_layout::DigitLayout;
use lru::LruCache;
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

pub struct Operator {
    handle: Arc<Handle>,
    schemes: Mutex<LruCache<SchemeKey, Scheme>>,
}

impl LayerNorm<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
            schemes: node.0.scheme_cache(SchemeDiversity::Low),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt_a, dt_w, d, .. } = args.meta()?;
        get_static!(d);

        let key = SchemeKey { dt_a, dt_w, d };
        self.schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?;
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
        let Meta { dt_a, dt_w, n, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            scale_layout,
            scale_base,
            bias_layout,
            bias_base,
            epsilon,
        } = args;
        let &[nsy, dsy] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, dsx] = x_layout.strides() else {
            unreachable!()
        };
        let &[dss] = scale_layout.strides() else {
            unreachable!()
        };
        let &[dsb] = bias_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   d
            nsy dsy
            nsx dsx
                dss
                dsb
        }

        let unit = dt_a.nbytes() as isize;
        if dsy != unit
            || dsx != unit
            || dss != dt_w.nbytes() as isize
            || dsb != dt_w.nbytes() as isize
        {
            return Err(strides_not_support("").into());
        };
        let key = SchemeKey { dt_a, dt_w, d };
        let scheme = self
            .schemes
            .lock()
            .unwrap()
            .try_get_or_insert(key, || Scheme::new(&self.handle, key))?
            .clone();

        let nsy = (nsy / unit) as i32;
        let nsx = (nsx / unit) as i32;
        let params = cuda::params![y_base, nsy, x_base, nsx, scale_base, bias_base, epsilon];

        scheme.module.launch(
            &scheme.name,
            n as u32,
            match scheme.ty {
                SchemeType::Padding => d,
                SchemeType::Folding { block_size } => block_size,
            } as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    dt_a: DigitLayout,
    dt_w: DigitLayout,
    d: usize,
}

#[derive(Clone)]
struct Scheme {
    ty: SchemeType,
    module: Arc<ModuleBox>,
    name: CString,
}

#[derive(Clone)]
enum SchemeType {
    Padding,
    Folding { block_size: usize },
}

impl Scheme {
    pub fn new(
        handle: &Arc<Handle>,
        SchemeKey { dt_a, dt_w, d }: SchemeKey,
    ) -> Result<Self, SchemeError> {
        let device = handle.device();
        let cc = device.compute_capability();
        let block_size = device.block_limit().max_threads;

        let ta = dt_name(dt_a);
        let tw = dt_name(dt_w);

        const CODE: &str = include_str!("layer_norm.cuh");
        if d <= block_size {
            let name = format!("layer_norm_{ta}_{tw}_padding_{d}");
            let module = handle.compile_kernel(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ s,
    {tw} const *__restrict__ b,
    float epsilon
){{
    padding<{d}>
    (y, stride_y, x, stride_x, s,b, epsilon);
}}"#
                )
            });

            Ok(Self {
                ty: SchemeType::Padding,
                module,
                name: CString::new(name).unwrap(),
            })
        } else {
            let n_threads_warp = device.warp_size();
            if d % n_threads_warp != 0 {
                Err(shape_not_support(format!(
                    "layernorm shape {d} must be multiple of warp size {n_threads_warp}"
                )))?
            }
            let max_num_warp_block = block_size / n_threads_warp;
            let to_divid = d / n_threads_warp;
            let num_warps_block = max_num_warp_block;
            let num_threads_block = n_threads_warp * num_warps_block;
            let num_items_thread = to_divid.div_ceil(num_warps_block);

            let name =
                format!("layer_norm_{ta}_{tw}_folding_{num_threads_block}x{num_items_thread}");
            let module = handle.compile_kernel(&name, cc, || {
                format!(
                    r#"{CODE}

extern "C" __global__ void {name}(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ s,
    {tw} const *__restrict__ b,
    float epsilon
){{
    folding<{num_threads_block}, {num_items_thread}>
   (y, stride_y, x, stride_x, s, b, epsilon, {d});
}}"#
                )
            });

            Ok(Self {
                ty: SchemeType::Folding { block_size },
                module,
                name: CString::new(name).unwrap(),
            })
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use core::f32;
    use digit_layout::{
        types::{F16, F32, F64},
        DigitLayout,
    };
    use std::ptr::null;

    fn dyn_args<H: Hardware>(dt_a: DigitLayout, dt_w: DigitLayout, d: usize) -> Args<H> {
        use std::ptr::null_mut;
        let yx_layout = TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]);
        let sb_layout = TensorLayout::new_dyn(dt_w, &[d.into()], &[dyn_()]);
        Args {
            y_layout: yx_layout.clone(),
            y_base: null_mut(),
            x_layout: yx_layout.clone(),
            x_base: null(),
            scale_layout: sb_layout.clone(),
            scale_base: null(),
            bias_layout: sb_layout.clone(),
            bias_base: null(),
            epsilon: 0.1f32,
        }
    }
    fn args<H: Hardware>(
        dt_a: DigitLayout,
        dt_w: DigitLayout,
        n: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        scale_base: *const H::Byte,
        bias_base: *const H::Byte,
        epsilon: f32,
    ) -> Args<H> {
        let yx_layout = TensorLayout::new_contiguous(dt_a, &[n, d]);
        let sb_layout = TensorLayout::new_contiguous(dt_w, &[d]);
        Args {
            y_layout: yx_layout.clone(),
            y_base,
            x_layout: yx_layout.clone(),
            x_base,
            scale_layout: sb_layout.clone(),
            scale_base,
            bias_layout: sb_layout.clone(),
            bias_base,
            epsilon,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            cuda::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use cuda::memcpy_d2h;
        use half::f16;
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        for k in 8..=13 {
            let n = 4;
            let d = 1 << k;
            let epsilon = 1.0f32;
            cpu_op.scheme(&dyn_args(F64, F64, d), 0).unwrap();
            gpu_op.scheme(&dyn_args(F16, F32, d), 0).unwrap();
            let y = vec![0.0f64; n * d];
            let mut x = vec![1.0f64; n * d];
            let mut scale = vec![1.0f64; d];
            let mut bias = vec![1.0f64; d];
            rand::rng().fill(&mut x[..]);
            rand::rng().fill(&mut scale[..]);
            rand::rng().fill(&mut bias[..]);
            let data_ans = gpu.apply(|ctx| {
                let stream = ctx.stream();
                #[cfg(use_nvidia)]
                let rt = &stream;
                #[cfg(use_iluvatar)]
                let rt = ctx;
                let mut data = rt.malloc::<f16>(y.len());
                let x = cast_load(&x, f16::from_f64, &stream);
                let scale = cast_load(&scale, |x| x as f32, &stream);
                let bias = cast_load(&bias, |x| x as f32, &stream);
                gpu_op
                    .launch(
                        &args(
                            F16,
                            F32,
                            n,
                            d,
                            data.as_mut_ptr().cast(),
                            x.as_ptr().cast(),
                            scale.as_ptr().cast(),
                            bias.as_ptr().cast(),
                            epsilon,
                        ),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; n * d];
                memcpy_d2h(&mut host, &data);
                host
            });

            let mut data_ref = y;
            cpu_op
                .launch(
                    &args(
                        F64,
                        F64,
                        n,
                        d,
                        data_ref.as_mut_ptr().cast(),
                        x.as_ptr().cast(),
                        scale.as_ptr().cast(),
                        bias.as_ptr().cast(),
                        epsilon,
                    ),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = data_ref
                .into_iter()
                .zip(data_ans)
                .map(|(a, b)| Diff::new(a as _, f16::to_f64(b)))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
