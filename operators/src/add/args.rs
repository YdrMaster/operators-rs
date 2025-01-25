use crate::{
    get_static, rank_mismatch, shape_mismatch, shape_not_support, utils::type_distinct, ConstPtr,
    Hardware, MutPtr, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;
use itertools::izip;
use std::{
    cmp::Ordering,
    ptr::{null, null_mut},
};

#[derive(Clone)]
pub struct Args<H: Hardware> {
    pub c_layout: TensorLayout,
    pub c_base: MutPtr<H>,
    pub a_layout: TensorLayout,
    pub a_base: ConstPtr<H>,
    pub b_layout: TensorLayout,
    pub b_base: ConstPtr<H>,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
    ) -> Self {
        Self {
            c_layout,
            c_base: null_mut(),
            a_layout,
            a_base: null(),
            b_layout,
            b_base: null(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Scheme(DigitLayout, Box<[isize]>);

impl Scheme {
    pub fn new<H: Hardware>(args: &Args<H>) -> Result<Self, SchemeError> {
        let Args {
            c_layout: c,
            a_layout: a,
            b_layout: b,
            ..
        } = args;
        // # 检查基本属性
        let dt = type_distinct(&[c.dt(), a.dt(), b.dt()])?;
        let ndim = c.ndim();
        if a.ndim() != ndim || b.ndim() != ndim {
            return Err(rank_mismatch(format!(
                "c.ndim = {}, a.ndim = {}, b.ndim = {}",
                c.ndim(),
                a.ndim(),
                b.ndim(),
            )));
        }
        // # 输入形状
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Dim {
            d: usize,
            c: isize,
            a: isize,
            b: isize,
        }
        let mut dims = Vec::with_capacity(ndim);
        for (&d, &da, &db, &sc, &sa, &sb) in izip!(
            c.shape(),
            a.shape(),
            b.shape(),
            c.strides(),
            a.strides(),
            b.strides()
        ) {
            get_static! {
                d  da db
                sc sa sb
            }
            if da != d || db != d {
                return Err(shape_mismatch(format!(
                    "c: {:?}, a: {:?}, b: {:?}",
                    c.shape(),
                    a.shape(),
                    b.shape(),
                )));
            }
            // 剔除初始的 1 长维度
            if d != 1 {
                if sc == 0 {
                    return Err(shape_not_support("Reducing is not allowed for add"));
                }
                dims.push(Dim {
                    d,
                    c: sc,
                    a: sa,
                    b: sb,
                })
            }
        }
        // # 排序
        dims.sort_unstable_by(|dim0, dim1| {
            let &Dim {
                d: d0,
                c: c0,
                a: a0,
                b: b0,
            } = dim0;
            let &Dim {
                d: d1,
                c: c1,
                a: a1,
                b: b1,
            } = dim1;
            use Ordering::Equal as Eq;
            match c0.abs().cmp(&c1.abs()) {
                Eq => match a0.abs().cmp(&a1.abs()) {
                    Eq => match b0.abs().cmp(&b1.abs()) {
                        Eq => d0.cmp(&d1),
                        ord => ord.reverse(),
                    },
                    ord => ord.reverse(),
                },
                ord => ord.reverse(),
            }
        });
        // # 合并连续维度
        let mut ndim = dims.len();
        for i in (1..dims.len()).rev() {
            let (head, tail) = dims.split_at_mut(i);
            let f = &mut head[i - 1]; // f for front
            let b = &mut tail[0]; // b for back
            let d = b.d as isize;
            if b.c * d == f.c && b.a * d == f.a && b.b * d == f.b {
                *f = Dim { d: b.d * f.d, ..*b };
                *b = Dim {
                    d: 1,
                    c: 0,
                    a: 0,
                    b: 0,
                };
                ndim -= 1
            }
        }
        // # 合并空间
        let mut layout = vec![0isize; 1 + ndim * 4].into_boxed_slice();
        {
            let (idx, tail) = layout.split_at_mut(1 + ndim);
            let (c_, tail) = tail.split_at_mut(ndim);
            let (a_, b_) = tail.split_at_mut(ndim);
            for (Dim { d, c, a, b }, idx, c_, a_, b_) in
                izip!(dims.into_iter().filter(|d| d.d != 1), &mut *idx, c_, a_, b_)
            {
                *idx = d as _;
                *c_ = c;
                *a_ = a;
                *b_ = b;
            }
            idx[ndim] = 1;
            for i in (1..=ndim).rev() {
                idx[i - 1] *= idx[i];
            }
        }
        Ok(Self(dt, layout))
    }

    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.0
    }

    /// 执行方案维数。
    #[inline]
    pub fn ndim(&self) -> usize {
        (self.1.len() - 1) / 4
    }

    /// 读写单元数量。
    #[inline]
    pub fn count(&self) -> usize {
        self.1[0] as _
    }

    /// 索引步长。
    #[inline]
    pub fn idx_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.1[1..][..ndim]
    }

    #[inline]
    pub fn c_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.1[1 + ndim..][..ndim]
    }

    #[inline]
    pub fn a_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.1[1 + ndim * 2..][..ndim]
    }

    #[inline]
    pub fn b_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.1[1 + ndim * 3..][..ndim]
    }
}
