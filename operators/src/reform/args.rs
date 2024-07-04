use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, Argument, ErrorPosition, Handle, TensorLayout};
use digit_layout::DigitLayout;
use std::iter::zip;

pub struct Args<H: Handle> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.dst_layout.dt();
        if self.src_layout.dt() != dt {
            return Err(locate_error!());
        }
        let ndim = self.dst_layout.ndim();
        if ndim < 2 || self.src_layout.ndim() != ndim {
            return Err(locate_error!());
        }
        for (&dst, &src) in zip(self.dst_layout.shape(), self.src_layout.shape()) {
            if Argument::merge(&[dst, src]).is_err() {
                return Err(locate_error!());
            }
        }
        Ok(Meta { dt })
    }
}

pub(super) struct Scheme(Vec<isize>);

impl Scheme {
    pub fn from<H: Handle>(args: &Args<H>) -> Result<Self, ErrorPosition> {
        let Args {
            dst_layout: dst_,
            src_layout: src_,
            ..
        } = args;

        let unit = dst_.dt().nbytes();
        if src_.dt().nbytes() != unit {
            return Err(locate_error!());
        }

        let ndim = dst_.ndim();
        if src_.ndim() != ndim {
            return Err(locate_error!());
        }
        // 检查形状
        let mut shape = vec![0isize; ndim];
        let mut dst = vec![0isize; ndim];
        let mut src = vec![0isize; ndim];
        {
            let dd = dst_.shape();
            let ds = src_.shape();
            let sd = dst_.strides();
            let ss = src_.strides();
            for i in 0..ndim {
                let d = *Argument::merge(&[dd[i], ds[i]]).map_err(|_| locate_error!())?;
                shape[i] = *d.get_static().ok_or_else(|| locate_error!())? as _;
                dst[i] = *sd[i].get_static().ok_or_else(|| locate_error!())?;
                src[i] = *ss[i].get_static().ok_or_else(|| locate_error!())?;
            }
        }
        // 合并连续维度
        let mut unit = unit as _;
        'out: loop {
            for i in 0..ndim {
                if dst[i] == unit && src[i] == unit {
                    unit *= shape[i];
                    shape[i] = 1;
                    dst[i] = 0;
                    src[i] = 0;
                    continue 'out;
                }
            }
            break;
        }
        // 移除无效维度
        let mut i = 0;
        while i < shape.len() {
            if shape[i] == 1 {
                shape.swap_remove(i);
                dst.swap_remove(i);
                src.swap_remove(i);
            } else {
                i += 1;
            }
        }
        // 合并空间
        let ndim = shape.len();
        let mut layout = vec![0isize; 1 + ndim * 3];
        layout[0] = unit as _;
        layout[1..][..ndim].copy_from_slice(&shape);
        layout[1 + ndim..][..ndim].copy_from_slice(&dst);
        layout[1 + ndim * 2..][..ndim].copy_from_slice(&src);

        Ok(Self(layout))
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        (self.0.len() - 1) / 3
    }

    #[inline]
    pub fn unit(&self) -> usize {
        self.0[0] as _
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        let ndim = self.ndim();
        unsafe { std::mem::transmute(&self.0[1..][..ndim]) }
    }

    #[inline]
    pub fn dst_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[1 + ndim..][..ndim]
    }

    #[inline]
    pub fn src_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[1 + ndim * 2..][..ndim]
    }
}

#[test]
fn test_scheme() {
    use crate::common_cpu::Handle as Cpu;
    use digit_layout::types::F16;
    use std::ptr::{null, null_mut};
    let args = Args::<Cpu> {
        dst_layout: TensorLayout::new(
            F16,
            &[1.into(), 2.into(), 3.into(), 4.into()],
            &[48.into(), 24.into(), 8.into(), 2.into()],
        ),
        dst_base: null_mut(),
        src_layout: TensorLayout::new(
            F16,
            &[1.into(), 2.into(), 3.into(), 4.into()],
            &[48.into(), 8.into(), 16.into(), 2.into()],
        ),
        src_base: null(),
    };
    let scheme = Scheme::from(&args).unwrap();
    println!("ndim = {}", scheme.ndim());
    println!("unit = {}", scheme.unit());
    println!("shape = {:?}", scheme.shape());
    println!("dst_strides = {:?}", scheme.dst_strides());
    println!("src_strides = {:?}", scheme.src_strides());
}
