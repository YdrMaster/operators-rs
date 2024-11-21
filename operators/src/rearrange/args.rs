use crate::{
    rank_mismatch, shape_mismatch, shape_not_support, static_from, utils::type_distinct, ConstPtr,
    Hardware, MutPtr, SchemeError, TensorLayout,
};
use std::{
    cmp::Ordering,
    iter::zip,
    ptr::{null, null_mut},
};

#[derive(Clone)]
pub struct Args<H: Hardware> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
}

impl<H: Hardware> Args<H> {
    pub fn new_null(dst_layout: TensorLayout, src_layout: TensorLayout) -> Self {
        Self {
            dst_layout,
            dst_base: null_mut(),
            src_layout,
            src_base: null(),
        }
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(super) struct Scheme(Vec<isize>);

impl Scheme {
    pub fn new<H: Hardware>(args: &Args<H>) -> Result<Self, SchemeError> {
        let Args {
            dst_layout: dst_,
            src_layout: src_,
            ..
        } = args;
        // # 检查基本属性
        let _ = type_distinct(&[dst_.dt(), src_.dt()])?;
        let ndim = dst_.ndim();
        if src_.ndim() != ndim {
            return Err(rank_mismatch(format!(
                "dst.ndim = {}, src.ndim = {}",
                dst_.ndim(),
                src_.ndim()
            )));
        }
        // # 输入形状
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Dim {
            len: usize,
            dst: isize,
            src: isize,
        }
        let mut dims = Vec::with_capacity(ndim);
        {
            let dd = dst_.shape();
            let ds = src_.shape();
            let sd = dst_.strides();
            let ss = src_.strides();
            for i in 0..ndim {
                let dd = *static_from(&dd[i])?;
                let ds = *static_from(&ds[i])?;
                if dd != ds {
                    Err(shape_mismatch(format!("dst[{i}] = {dd}, src[{i}] = {ds}")))?;
                }
                // 静态化
                let dim = Dim {
                    len: dd,
                    dst: *static_from(&sd[i])?,
                    src: *static_from(&ss[i])?,
                };
                // 剔除初始的 1 长维度
                if dim.len != 1 {
                    if dim.dst == 0 {
                        return Err(shape_not_support(
                            "Reducing is not allowed for rearrangement.",
                        ));
                    }
                    dims.push(dim);
                }
            }
        }
        // # 排序
        impl PartialOrd for Dim {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Dim {
            /// dst 绝对值降序 -> src 绝对值降序 -> len 升序
            fn cmp(&self, other: &Self) -> Ordering {
                use Ordering::Equal as Eq;
                match self.dst.abs().cmp(&other.dst.abs()) {
                    Eq => match self.src.abs().cmp(&other.src.abs()) {
                        Eq => self.len.cmp(&other.len),
                        neq => neq.reverse(),
                    },
                    neq => neq.reverse(),
                }
            }
        }
        dims.sort_unstable();
        // # 合并连续维度
        let mut unit = dst_.dt().nbytes() as isize;
        let mut ndim = dims.len();
        // ## 合并末尾连续维度到 unit
        for dim in dims.iter_mut().rev() {
            if dim.dst == unit && dim.src == unit {
                unit *= dim.len as isize;
                ndim -= 1;
            } else {
                break;
            }
        }
        dims.truncate(ndim);
        // ## 合并任意连续维度
        for i in (1..dims.len()).rev() {
            let (head, tail) = dims.split_at_mut(i);
            let f = &mut head[i - 1]; // f for front
            let b = &mut tail[0]; // b for back
            let len = b.len as isize;
            if b.dst * len == f.dst && b.src * len == f.src {
                *f = Dim {
                    len: b.len * f.len,
                    dst: b.dst,
                    src: b.src,
                };
                *b = Dim {
                    len: 1,
                    dst: 0,
                    src: 0,
                };
                ndim -= 1;
            }
        }
        // # 合并空间
        let mut layout = vec![0isize; 2 + ndim * 3];
        layout[0] = unit as _;
        layout[ndim + 1] = 1;
        for (i, Dim { len, dst, src }) in dims.into_iter().filter(|d| d.len != 1).enumerate() {
            layout[1 + i] = len as _;
            layout[2 + ndim + i] = dst;
            layout[2 + ndim * 2 + i] = src;
        }
        for i in (1..=ndim).rev() {
            layout[i] *= layout[i + 1];
        }
        Ok(Self(layout))
    }

    /// 拆分 unit 到更小的规模以利于并行
    #[allow(dead_code)]
    pub fn distribute_unit(&self, candidates: impl IntoIterator<Item = usize>) -> Self {
        let unit = candidates
            .into_iter()
            .find(|n| self.unit() % n == 0)
            .unwrap();
        if unit == self.unit() {
            return Self(self.0.clone());
        }

        let ndim = self.ndim();
        let mut layout = vec![0isize; 2 + (ndim + 1) * 3];
        layout[0] = unit as _;

        let (_, tail) = layout.split_at_mut(1);
        let (idx, tail) = tail.split_at_mut(ndim + 2);
        let (dst, src) = tail.split_at_mut(ndim + 1);

        let (_, tail) = self.0.split_at(1);
        let (idx_, tail) = tail.split_at(ndim + 1);
        let (dst_, src_) = tail.split_at(ndim);

        idx[ndim + 1] = 1;
        let extra = (self.unit() / unit) as isize;
        for (new, old) in zip(idx, idx_) {
            *new = *old * extra;
        }

        fn copy_value(new: &mut [isize], old: &[isize], unit: usize) {
            let [head @ .., tail] = new else {
                unreachable!()
            };
            head.copy_from_slice(old);
            *tail = unit as _;
        }
        copy_value(dst, dst_, unit);
        copy_value(src, src_, unit);

        Self(layout)
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        (self.0.len() - 2) / 3
    }

    #[inline]
    pub fn unit(&self) -> usize {
        self.0[0] as _
    }

    #[allow(unused)]
    #[inline]
    pub fn count(&self) -> usize {
        self.0[1] as _
    }

    #[allow(unused)]
    #[inline]
    pub fn idx_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[2..][..ndim]
    }

    #[inline]
    pub fn dst_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[2 + ndim..][..ndim]
    }

    #[inline]
    pub fn src_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[2 + ndim * 2..][..ndim]
    }

    #[allow(dead_code)]
    #[inline]
    pub fn shape(&self) -> impl Iterator<Item = usize> + '_ {
        let ndim = self.ndim();
        self.0[1..][..ndim + 1]
            .windows(2)
            .map(|pair| (pair[0] / pair[1]) as usize)
    }
}

#[test]
fn test_scheme() {
    use crate::common_cpu::Cpu;
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};

    {
        let shape = [4, 3, 2, 1, 2, 3, 4];
        let args = Args::<Cpu> {
            dst_layout: TensorLayout::new(F16, &shape, &[288, 96, 48, 48, 24, 8, 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new(F16, &shape, &[576, 192, 96, 48, 8, 16, 2]),
            src_base: null(),
        };
        let scheme = Scheme::new(&args).unwrap();
        assert_eq!(scheme.ndim(), 3);
        assert_eq!(scheme.unit(), 8);
        assert_eq!(scheme.count(), 24 * 2 * 3);
        assert_eq!(scheme.idx_strides(), [6, 3, 1]);
        assert_eq!(scheme.dst_strides(), [48, 24, 8]);
        assert_eq!(scheme.src_strides(), [96, 8, 16]);
        assert_eq!(scheme.shape().collect::<Vec<_>>(), [24, 2, 3]);
    }
    {
        let shape = [32, 2, 32, 456, 128];
        let args = Args::<Cpu> {
            dst_layout: TensorLayout::new(
                F16,
                &shape,
                &[33554432 * 2, 16777216 * 2, 524288 * 2, 128 * 2, 1 * 2],
            ),
            dst_base: null_mut(),
            src_layout: TensorLayout::new(
                F16,
                &shape,
                &[33554432 * 2, 16777216 * 2, 524288 * 2, 128 * 2, 1 * 2],
            ),
            src_base: null(),
        };
        let scheme = Scheme::new(&args).unwrap();
        #[rustfmt::skip]
        assert_eq!(
            scheme.0,
            [
                116736,
                2048, 1,
                1048576,
                1048576,
            ]
        );
        let scheme = scheme.distribute_unit((0..=5).rev().map(|n| 32 * (1 << n)));
        #[rustfmt::skip]
        assert_eq!(
            scheme.0,
            [
                1024,
                116736 / 1024 * 2048, 116736 / 1024, 1,
                1048576, 1024,
                1048576, 1024,
            ]
        );
    }
}
