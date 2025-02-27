﻿use super::ReduceOp;
use crate::{
    rearrange, shape_mismatch, strides_not_support, utils::type_distinct, Hardware, LaunchError,
};
use digit_layout::DigitLayout;

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
    pub(super) fn meta(&self) -> Result<Meta, LaunchError> {
        let Self {
            pair:
                rearrange::Args {
                    dst_layout,
                    src_layout,
                    ..
                },
            ..
        } = self;

        let dt = type_distinct(&[dst_layout.dt, src_layout.dt])?;

        let dst = &dst_layout.layout;
        let &[dst] = dst
            .merge_be(0, dst.ndim())
            .ok_or(strides_not_support(""))?
            .shape()
        else {
            unreachable!()
        };

        let src = &src_layout.layout;
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
