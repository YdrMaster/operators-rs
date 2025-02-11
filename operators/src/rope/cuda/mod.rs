use super::{args::Meta, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    get_static, shape_not_support, strides_not_support, type_not_support, Blob, ByteOf,
    LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::{types as ty, DigitLayout};
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
    module: Arc<ModuleBox>,
}

const MODULE: &str = "rope";
const POS_U32: &str = "rope_f16_u32";
const POS_U64: &str = "rope_f16_u64";

impl Rope<Gpu> for Operator {
    fn build_sincos<QA>(
        _dt: DigitLayout,
        _nctx: usize,
        _dh: usize,
        queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        SinCosTable {
            nctx: 0,
            mem: queue_alloc.alloc(0),
        }
    }

    fn build_pos<I, QA>(
        dt: digit_layout::DigitLayout,
        nt: usize,
        iter: I,
        queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut host = Blob::new(dt.nbytes() * nt);
        match dt {
            ty::U32 => fill_pos(host.as_mut_ptr().cast::<u32>(), nt, iter),
            ty::U64 => fill_pos(host.as_mut_ptr().cast::<u64>(), nt, iter),
            _ => todo!(),
        }

        let mut blob = queue_alloc.alloc(host.len());
        queue_alloc.queue().memcpy_h2d(&mut blob, &host);
        blob
    }
}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        let cc = node.0.device().compute_capability();
        Self {
            _handle: node.0.clone(),
            max_threads_block: node.0.device().block_limit().max_threads,
            module: node.0.compile_kernel(MODULE, cc, format_code),
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
        let Meta {
            dt_t, dt_p, nt, dh, ..
        } = args.meta()?;

        if dt_t != ty::F16 {
            return Err(type_not_support("").into());
        }
        let name = match dt_p {
            ty::U32 => POS_U32,
            ty::U64 => POS_U64,
            _ => return Err(type_not_support("").into()),
        };

        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            ..
        } = args;
        let &[_, nh, _] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        get_static! {
            nt nh dh
            st sh sd
            sp
        }

        let unit = dt_t.nbytes() as isize;
        if sd != unit || sp != dt_p.nbytes() as isize {
            return Err(strides_not_support("").into());
        }

        let dh = dh / 2;
        let st = (st / unit / 2) as i32;
        let sh = (sh / unit / 2) as i32;
        let params = cuda::params![t_base, st, sh, p_base, theta];

        if self.max_threads_block % dh != 0 {
            return Err(shape_not_support("").into());
        }

        let max_nh_l = (self.max_threads_block / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;

        self.module.launch(
            CString::new(name).unwrap(),
            (nt as _, nh_h as _),
            (nh_l as _, dh as _),
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn format_code() -> String {
    const CODE: &str = include_str!("rope.cuh");
    format!(
        r#"{CODE}

extern "C" __global__ void {POS_U32}(
    half2 *__restrict__ t,
    int const stride_token,
    int const stride_head,
    unsigned int const *__restrict__ pos,
    float theta
){{
    padding(t, stride_token, stride_head, pos, theta);
}}

extern "C" __global__ void {POS_U64}(
    half2 *__restrict__ t,
    int const stride_token,
    int const stride_head,
    unsigned long long const *__restrict__ pos,
    float theta
){{
    padding(t, stride_token, stride_head, pos, theta);
}}"#
    )
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator, POS_U32, POS_U64};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64, U32},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt_t: DigitLayout, dt_p: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            t_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 3], &[dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new_dyn(dt_p, &[dyn_()], &[dyn_()]),
            p_base: null(),
            sin_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            sin_base: null(),
            cos_layout: TensorLayout::new_dyn(dt_t, &[dyn_(); 2], &[dyn_(); 2]),
            cos_base: null(),
            theta: 0.,
        }
    }

    fn args<H: Hardware>(
        dt_t: DigitLayout,
        dt_p: DigitLayout,
        nt: usize,
        nh: usize,
        dh: usize,
        theta: f32,
        t_base: *mut H::Byte,
        p_base: *const H::Byte,
    ) -> Args<H> {
        use std::ptr::null;
        Args {
            t_layout: TensorLayout::new_contiguous(dt_t, &[nt, nh, dh]),
            t_base,
            p_layout: TensorLayout::new_contiguous(dt_p, &[nt]),
            p_base,
            sin_layout: TensorLayout::new_contiguous(dt_t, &[0, dh]),
            sin_base: null(),
            cos_layout: TensorLayout::new_contiguous(dt_t, &[0, dh]),
            cos_base: null(),
            theta,
        }
    }

    #[test]
    fn test_compile() {
        use std::ffi::CString;

        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        op.scheme(&dyn_args(F16, U32), 0).unwrap();

        gpu.apply(|ctx| {
            println!(
                "{POS_U32}\n{}",
                op.module.load(CString::new(POS_U32).unwrap(), ctx).info()
            );
            println!(
                "{POS_U64}\n{}",
                op.module.load(CString::new(POS_U64).unwrap(), ctx).info()
            );
        })
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
        cpu_op.scheme(&dyn_args(F64, U32), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16, U32), 0).unwrap();

        const NT: usize = 7;
        let nh = 32;
        let dh = 64;

        let mut t = vec![0.0f64; NT * nh * dh];
        rand::rng().fill(&mut t[..]);
        let p: [u32; NT] = [0, 1, 2, 3, 7, 8, 1];

        let t_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            #[cfg(use_nvidia)]
            let rt = &stream;
            #[cfg(use_iluvatar)]
            let rt = ctx;
            let mut t = cast_load(&t, f16::from_f64, &stream);

            let p = rt.from_host(&p);
            gpu_op
                .launch(
                    &args(
                        F16,
                        U32,
                        NT,
                        nh,
                        dh,
                        1e4,
                        t.as_mut_ptr().cast(),
                        p.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; NT * nh * dh];
            memcpy_d2h(&mut host, &t);
            host
        });

        let mut t_ref = t;
        cpu_op
            .launch(
                &args(
                    F64,
                    U32,
                    NT,
                    nh,
                    dh,
                    1e4,
                    t_ref.as_mut_ptr().cast(),
                    p.as_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = t_ref
            .into_iter()
            .zip(t_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
