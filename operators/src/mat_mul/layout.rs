use common::{locate_error, DataLayout, ErrorPosition, TensorLayout};
use std::mem::swap;

pub struct LayoutAttrs {
    pub c: TensorLayout,
    pub a: TensorLayout,
    pub b: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub c_stride: isize,
    pub c_offset: usize,
    pub c_ld: isize,
    pub ab_swap: bool,

    pub a_stride: isize,
    pub a_offset: usize,
    pub a_ld: isize,
    pub a_trans: bool,

    pub b_stride: isize,
    pub b_offset: usize,
    pub b_ld: isize,
    pub b_trans: bool,
}

impl SchemeLayout {
    pub fn new(
        dt: DataLayout,
        LayoutAttrs { c, a, b }: LayoutAttrs,
    ) -> Result<Self, ErrorPosition> {
        if c.dt() != dt || a.dt() != dt || b.dt() != dt {
            return Err(locate_error!("Inconsistent data types"));
        }
        // 确认矩阵结构匹配
        let mut c = Matrix::try_from(&c)?;
        let mut a = Matrix::try_from(&a)?;
        let mut b = Matrix::try_from(&b)?;
        if c.r != a.r || c.c != b.c || a.c != b.r {
            return Err(locate_error!("Inconsistent matrix shapes"));
        }
        // 确认批处理结构匹配
        let batch = c.batch;
        if !a.match_batch(batch) || !b.match_batch(batch) {
            return Err(locate_error!("Inconsistent batch sizes"));
        }
        // 确认 c 列优先
        let ab_swap = if c.rs == 1 {
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
            return Err(locate_error!("Matrix is not contiguous"));
        };
        macro_rules! trans {
            ($m:expr) => {
                if $m.rs == 1 {
                    ($m.cs, false)
                } else if a.cs == 1 {
                    ($m.rs, true)
                } else {
                    return Err(locate_error!("Matrix is not contiguous"));
                }
            };
        }
        let (a_ld, a_transpose) = trans!(a);
        let (b_ld, b_transpose) = trans!(b);
        Ok(Self {
            batch,
            m: c.r,
            n: c.c,
            k: a.c,

            c_stride: c.stride,
            c_offset: c.offset,
            c_ld: c.cs,
            ab_swap,

            a_stride: a.stride,
            a_offset: a.offset,
            a_ld,
            a_trans: a_transpose,

            b_stride: b.stride,
            b_offset: b.offset,
            b_ld,
            b_trans: b_transpose,
        })
    }
}

#[derive(Clone, Debug)]
struct Matrix {
    pub batch: usize,
    pub stride: isize,
    pub r: usize,
    pub c: usize,
    pub rs: isize,
    pub cs: isize,
    pub offset: usize,
}

impl TryFrom<&TensorLayout> for Matrix {
    type Error = ErrorPosition;

    fn try_from(tensor: &TensorLayout) -> Result<Self, Self::Error> {
        let [batch @ .., r, c] = tensor.shape() else {
            return Err(locate_error!("Invalid matrix shape"));
        };
        let [stride @ .., rs, cs] = tensor.strides() else {
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
            _ => return Err(locate_error!("Invalid matrix shape")),
        };
        Ok(Self {
            batch,
            stride,
            r: *r,
            c: *c,
            rs: rs / unit,
            cs: cs / unit,
            offset: tensor.offset(),
        })
    }
}

impl Matrix {
    #[inline(always)]
    fn match_batch(&self, batch: usize) -> bool {
        self.batch == 1 || self.batch == batch
    }
    #[inline(always)]
    fn transpose(&mut self) {
        swap(&mut self.r, &mut self.c);
        swap(&mut self.rs, &mut self.cs);
    }
}
