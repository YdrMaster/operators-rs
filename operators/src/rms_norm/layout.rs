use common::{locate_error, DataLayout, ErrorPosition, TensorLayout};

pub struct LayoutAttrs {
    pub y: TensorLayout,
    pub x: TensorLayout,
    pub w: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub n: usize,
    pub d: usize,
    pub stride_y: isize,
    pub stride_x: isize,
    pub offset_y: usize,
    pub offset_x: usize,
    pub offset_w: usize,
}

impl SchemeLayout {
    pub fn new(
        dt: DataLayout,
        LayoutAttrs { y, x, w }: LayoutAttrs,
    ) -> Result<Self, ErrorPosition> {
        if y.dt() != dt {
            return Err(locate_error!());
        }
        if x.dt() != dt {
            return Err(locate_error!());
        }
        if w.dt() != dt {
            return Err(locate_error!());
        }
        let &[yn, yd] = y.shape() else {
            return Err(locate_error!());
        };
        let &[xn, xd] = x.shape() else {
            return Err(locate_error!());
        };
        let &[wd] = w.shape() else {
            return Err(locate_error!());
        };
        if yn != xn || yd != xd || yd != wd {
            return Err(locate_error!());
        }
        let &[yns, yds] = y.strides() else {
            unreachable!();
        };
        let &[xns, xds] = x.strides() else {
            unreachable!();
        };
        let &[wds] = w.strides() else {
            unreachable!();
        };
        let unit = dt.layout().size() as isize;
        if yds != unit || xds != unit || wds != unit {
            return Err(locate_error!());
        }

        Ok(Self {
            n: yn,
            d: yd,
            stride_y: yns / unit,
            stride_x: xns / unit,
            offset_y: y.offset(),
            offset_x: x.offset(),
            offset_w: w.offset(),
        })
    }
}
