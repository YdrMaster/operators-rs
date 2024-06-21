use common::{locate_error, ErrorPosition, TensorLayout};
use digit_layout::{types::U32, DigitLayout};

pub struct LayoutAttrs {
    pub t: TensorLayout,
    pub pos: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub n: usize,
    pub nh: usize,
    pub dh: usize,
    pub stride_token: isize,
    pub stride_head: isize,
    pub offset_t: usize,
    pub offset_pos: usize,
}

impl SchemeLayout {
    pub fn new(
        dt: DigitLayout,
        LayoutAttrs { t, pos }: LayoutAttrs,
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
        let unit = dt.nbytes() as isize;
        if dhs != unit || nps != U32.nbytes() as isize {
            return Err(locate_error!());
        }

        Ok(Self {
            n: nt,
            nh,
            dh,
            stride_token: nts / unit,
            stride_head: nhs / unit,
            offset_t: t.offset(),
            offset_pos: pos.offset(),
        })
    }
}
