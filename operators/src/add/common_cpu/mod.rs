use super::{args::Scheme, Add, Args};
use crate::{common_cpu::Cpu, ByteOf, LaunchError, QueueAlloc, SchemeError};
use digit_layout::types as ty;
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Operator;

impl Add<Cpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_node: &Self::TopoNode) -> Self {
        Self
    }
    #[inline]
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
        let scheme = Scheme::new(args)?;
        let c = args.c_base as isize;
        let a = args.a_base as isize;
        let b = args.b_base as isize;
        let idx_strides = scheme.idx_strides();
        let c_strides = scheme.c_strides();
        let a_strides = scheme.a_strides();
        let b_strides = scheme.b_strides();
        (0..scheme.count() as isize)
            .into_par_iter()
            .for_each(|mut rem| {
                let mut c = c;
                let mut a = a;
                let mut b = b;
                for (i, &s) in idx_strides.iter().enumerate() {
                    let k = rem / s;
                    c += k * c_strides[i];
                    a += k * a_strides[i];
                    b += k * b_strides[i];
                    rem %= s;
                }
                match scheme.dt() {
                    ty::F16 => add::<f16>(c, a, b),
                    ty::F32 => add::<f32>(c, a, b),
                    ty::F64 => add::<f64>(c, a, b),
                    _ => todo!(),
                }
            });
        Ok(())
    }
}

fn add<T: std::ops::Add<Output = T>>(c: isize, a: isize, b: isize) {
    let c = c as *mut T;
    let a = a as *const T;
    let b = b as *const T;
    unsafe { *c = a.read() + b.read() }
}
