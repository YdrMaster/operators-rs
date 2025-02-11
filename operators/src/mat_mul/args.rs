use crate::{
    dyn_not_support, rank_not_support, shape_mismatch, shape_not_support, strides_not_support,
    utils::type_distinct, ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;
use std::{
    mem::swap,
    ptr::{null, null_mut},
};

pub struct Args<H: Hardware> {
    pub c_layout: TensorLayout,
    pub c_base: MutPtr<H>,
    pub beta: f32,
    pub a_layout: TensorLayout,
    pub a_base: ConstPtr<H>,
    pub b_layout: TensorLayout,
    pub b_base: ConstPtr<H>,
    pub alpha: f32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct SchemeLayout {
    pub dt: DigitLayout,
    pub ab_swap: bool,
    pub a_trans: bool,
    pub b_trans: bool,

    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub c_stride: isize,
    pub c_ld: isize,

    pub a_stride: isize,
    pub a_ld: isize,

    pub b_stride: isize,
    pub b_ld: isize,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(
        c_layout: TensorLayout,
        beta: f32,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        alpha: f32,
    ) -> Self {
        Self {
            c_layout,
            c_base: null_mut(),
            beta,
            a_layout,
            a_base: null(),
            b_layout,
            b_base: null(),
            alpha,
        }
    }

    pub(super) fn layout(&self) -> Result<SchemeLayout, SchemeError> {
        let Self {
            c_layout,
            a_layout,
            b_layout,
            ..
        } = self;

        // 确认矩阵结构匹配
        let mut c = Matrix::try_from(&self.c_layout)?;
        let mut a = Matrix::try_from(&self.a_layout)?;
        let mut b = Matrix::try_from(&self.b_layout)?;
        if c.r != a.r || c.c != b.c || a.c != b.r {
            return Err(shape_mismatch("Inconsistent matrix shapes"));
        }
        // 确认批处理结构匹配
        let batch = c.batch;
        if !a.match_batch(batch) || !b.match_batch(batch) {
            return Err(shape_mismatch("Inconsistent batch sizes"));
        }
        // 确认 c 列优先
        let ab_swap = if c.rs == 1 && c.cs != 1 {
            // Nothing to do
            false
        } else if c.cs == 1 {
            // cT = bT.aT
            c.transpose();
            a.transpose();
            b.transpose();
            swap(&mut a, &mut b);
            true
        } else {
            return Err(strides_not_support("Matrix is not contiguous"));
        };

        let (a_ld, a_trans) = a.ld_trans()?;
        let (b_ld, b_trans) = b.ld_trans()?;
        Ok(SchemeLayout {
            dt: type_distinct(&[c_layout.dt(), a_layout.dt(), b_layout.dt()])?,
            ab_swap,
            a_trans,
            b_trans,

            batch,
            m: c.r,
            n: c.c,
            k: a.c,

            c_stride: c.stride,
            c_ld: c.cs,

            a_stride: a.stride,
            a_ld,

            b_stride: b.stride,
            b_ld,
        })
    }
}

#[derive(Clone, Debug)]
struct Matrix {
    batch: usize,
    stride: isize,
    r: usize,
    c: usize,
    rs: isize,
    cs: isize,
}

impl TryFrom<&TensorLayout> for Matrix {
    type Error = SchemeError;

    fn try_from(tensor: &TensorLayout) -> Result<Self, Self::Error> {
        let Some(shape) = MaybeDyn::get_all(tensor.shape()) else {
            return Err(dyn_not_support(""));
        };
        let Some(strides) = MaybeDyn::get_all(tensor.strides()) else {
            return Err(dyn_not_support(""));
        };

        let [batch @ .., r, c] = shape else {
            return Err(rank_not_support("Matrix must have rank 2 or more"));
        };
        let [stride @ .., rs, cs] = strides else {
            unreachable!();
        };
        let unit = tensor.dt().nbytes() as isize;
        let (batch, stride) = match batch {
            [] | [1] => {
                assert!(matches!(stride, [] | [_]));
                (1, 0)
            }
            &[batch] => {
                let &[stride] = stride else { unreachable!() };
                (batch, stride / unit)
            }
            _ => return Err(shape_not_support("Higher-rank tensors not supported")),
        };
        Ok(Self {
            batch,
            stride,
            r: *r,
            c: *c,
            rs: rs / unit,
            cs: cs / unit,
        })
    }
}

impl Matrix {
    #[inline(always)]
    fn match_batch(&self, batch: usize) -> bool {
        self.batch == 1 || self.batch == batch
    }
    #[inline(always)]
    fn ld_trans(&mut self) -> Result<(isize, bool), SchemeError> {
        match (self.rs, self.cs) {
            (1, cs) => Ok((cs, false)),
            (rs, 1) => Ok((rs, true)),
            (_, _) => Err(strides_not_support("Matrix is not contiguous")),
        }
    }
    #[inline(always)]
    fn transpose(&mut self) {
        swap(&mut self.r, &mut self.c);
        swap(&mut self.rs, &mut self.cs);
    }
}
