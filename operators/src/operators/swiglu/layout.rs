use crate::{locate_error, DataLayout, ErrorPosition, TensorLayout};

pub struct KnTensorLayout {
    pub gate: TensorLayout,
    pub up: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub n: usize,
    pub d: usize,
    pub stride_gate: isize,
    pub stride_up: isize,
}

impl SchemeLayout {
    pub fn new(
        dt: DataLayout,
        KnTensorLayout { gate, up }: KnTensorLayout,
    ) -> Result<Self, ErrorPosition> {
        if gate.dt() != dt || up.dt() != dt || dt.packed() != 1 {
            return Err(locate_error!());
        }
        let &[gn, gd] = gate.shape() else {
            return Err(locate_error!());
        };
        let &[un, ud] = up.shape() else {
            return Err(locate_error!());
        };
        if gn != un || gd != ud {
            return Err(locate_error!());
        }
        let &[gns, gds] = gate.strides() else {
            unreachable!();
        };
        let &[uns, uds] = up.strides() else {
            unreachable!();
        };
        let unit = dt.layout().size() as isize;
        if gds != unit || uds != unit {
            return Err(locate_error!());
        }

        Ok(Self {
            n: gn,
            d: gd,
            stride_gate: gns / unit,
            stride_up: uns / unit,
        })
    }
}
