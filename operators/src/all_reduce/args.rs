use super::ReduceOp;
use crate::{
    dyn_not_support, rearrange, shape_mismatch, strides_not_support, utils::type_distinct,
    Hardware, MaybeDyn, SchemeError,
};
use digit_layout::DigitLayout;
use ndarray_layout::ArrayLayout;

pub struct Args<H: Hardware> {
    pub pair: rearrange::Args<H>,
    pub op: ReduceOp,
}

impl<H: Hardware> AsRef<rearrange::Args<H>> for Args<H> {
    #[inline]
    fn as_ref(&self) -> &rearrange::Args<H> {
        &self.pair
    }
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub size: usize,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            pair:
                rearrange::Args {
                    dst_layout,
                    src_layout,
                    ..
                },
            ..
        } = self;

        let dt = type_distinct(&[dst_layout.dt(), src_layout.dt()])?;

        let Some(shape) = MaybeDyn::get_all(dst_layout.shape()) else {
            return Err(dyn_not_support(""));
        };
        let Some(strides) = MaybeDyn::get_all(dst_layout.strides()) else {
            return Err(dyn_not_support(""));
        };
        let dst = ArrayLayout::<2>::new(shape, strides, 0);
        let &[dst] = dst
            .merge_be(0, dst.ndim())
            .ok_or(strides_not_support(""))?
            .shape()
        else {
            unreachable!()
        };

        let Some(shape) = MaybeDyn::get_all(src_layout.shape()) else {
            return Err(dyn_not_support(""));
        };
        let Some(strides) = MaybeDyn::get_all(src_layout.strides()) else {
            return Err(dyn_not_support(""));
        };
        let src = ArrayLayout::<2>::new(shape, strides, 0);
        let &[src] = src
            .merge_be(0, src.ndim())
            .ok_or(strides_not_support(""))?
            .shape()
        else {
            unreachable!()
        };

        if dst != src {
            return Err(shape_mismatch(""));
        }

        Ok(Meta { dt, size: dst })
    }
}
