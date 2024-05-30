use crate::{locate_error, DataLayout, ErrorPosition, TensorLayout};

pub struct RmsNormTensorLayout {
    pub y: TensorLayout,
    pub x: TensorLayout,
    pub w: TensorLayout,
}

pub(super) struct RmsNormSchemeLayout {
    pub n: usize,
    pub d: usize,
    pub stride_y: isize,
    pub stride_x: isize,
}

impl RmsNormSchemeLayout {
    pub fn new(
        dt: DataLayout,
        RmsNormTensorLayout { y, x, w }: RmsNormTensorLayout,
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

        Ok(RmsNormSchemeLayout {
            n: yn,
            d: yd,
            stride_y: yns,
            stride_x: xns,
        })
    }
}
