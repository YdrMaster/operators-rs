use super::{layout::SchemeLayout, LayoutAttrs, MatMul, Params};
use common::{locate_error, ErrorPosition, QueueOf};
use dev_common_cpu::Device as Cpu;
use digit_layout::{types::F16, DigitLayout};
use half::f16;

pub struct Operator {
    dt: DigitLayout,
}

impl common::Operator for Operator {
    type Handle = Cpu;

    type Config = DigitLayout;
    type Error = ErrorPosition;
    #[inline]
    fn new(config: &Self::Config) -> Result<Self, Self::Error> {
        if *config == F16 {
            Ok(Self { dt: *config })
        } else {
            Err(locate_error!())
        }
    }
}

pub struct Scheme(SchemeLayout);

impl MatMul<Cpu> for Scheme {}

impl common::Scheme for Scheme {
    type Device = Cpu;
    type Operator = Operator;

    type LayoutAttrs = LayoutAttrs;
    type Error = ErrorPosition;
    #[inline]
    fn new(op: &Operator, layout: Self::LayoutAttrs) -> Result<Self, Self::Error> {
        SchemeLayout::new(op.dt, layout).map(Self)
    }

    type Params = Params<Cpu>;
    fn launch(&self, params: &Self::Params, _queue: &QueueOf<Cpu>) {
        let SchemeLayout {
            batch,
            m,
            n,
            k,

            c_stride,
            c_offset,
            c_ld,
            ab_swap,

            a_stride,
            a_offset,
            a_ld,
            a_trans,

            b_stride,
            b_offset,
            b_ld,
            b_trans,
        } = self.0;
        let &(c, beta, a, b, alpha) = params;
        let (a, b) = if ab_swap { (b, a) } else { (a, b) };
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        let c = unsafe { c.add(c_offset) }.cast::<f16>();
        let a = unsafe { a.add(a_offset) }.cast::<f16>();
        let b = unsafe { b.add(b_offset) }.cast::<f16>();

        for i in 0..batch as isize {
            unsafe {
                let c = c.offset(i * c_stride);
                let a = a.offset(i * a_stride);
                let b = b.offset(i * b_stride);
                gemm::gemm(
                    m,
                    n,
                    k,
                    c,
                    c_ld,
                    1,
                    beta != 0.,
                    a,
                    lhs_cs,
                    lhs_rs,
                    b,
                    rhs_cs,
                    rhs_rs,
                    f16::from_f32(beta),
                    f16::from_f32(alpha),
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                )
            }
        }
    }
}
