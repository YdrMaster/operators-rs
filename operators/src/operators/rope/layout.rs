use crate::{locate_error, DataLayout, ErrorPosition, TensorLayout, U32};

pub struct KnTensorLayout {
    pub t: TensorLayout,
    pub pos: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub n: usize,
    pub nh: usize,
    pub dh: usize,
    pub stride_token: isize,
    pub stride_head: isize,
}

impl SchemeLayout {
    pub fn new(
        dt: DataLayout,
        KnTensorLayout { t, pos }: KnTensorLayout,
    ) -> Result<Self, ErrorPosition> {
        if t.dt() != dt || dt.packed() != 1 {
            return Err(locate_error!());
        }
        if pos.dt() != U32 {
            return Err(locate_error!());
        }
        let &[nt, nh, dh] = t.shape() else {
            return Err(locate_error!());
        };
        let &[np] = pos.shape() else {
            return Err(locate_error!());
        };
        if nt != np || dh % 2 != 0 {
            return Err(locate_error!());
        }
        let &[nts, nhs, dhs] = t.strides() else {
            unreachable!();
        };
        let &[nps] = pos.strides() else {
            unreachable!();
        };
        let unit = dt.layout().size() as isize;
        if dhs != unit || nps != U32.layout().size() as isize {
            return Err(locate_error!());
        }

        Ok(Self {
            n: nt,
            nh,
            dh,
            stride_token: nts / unit,
            stride_head: nhs / unit,
        })
    }
}
