use crate::{
    dyn_not_support, shape_mismatch, strides_not_support,
    utils::{sizeof, type_distinct},
    ConstPtr, Hardware, MaybeDyn, MutPtr, SchemeError, TensorLayout,
};
use ndarray_layout::ArrayLayout;

pub struct Args<H: Hardware> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
    pub root: usize,
}

pub(super) struct Meta {
    pub size: usize,
}

impl<H: Hardware> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, SchemeError> {
        let Self {
            dst_layout,
            src_layout,
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
            .merge(0..dst.ndim())
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
            .merge(0..src.ndim())
            .ok_or(strides_not_support(""))?
            .shape()
        else {
            unreachable!()
        };

        if dst != src {
            return Err(shape_mismatch(""));
        }

        Ok(Meta {
            size: dst * sizeof(dt)?,
        })
    }
}
