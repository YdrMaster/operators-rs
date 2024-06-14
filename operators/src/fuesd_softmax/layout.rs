use common::{locate_error, DataLayout, ErrorPosition, TensorLayout};

pub struct LayoutAttrs {
    pub att: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub nh: usize,
    pub seq_len: usize,
    pub att_len: usize,
    pub stride_head: isize,
    pub stride_token: isize,
    pub offset: usize,
}

impl SchemeLayout {
    pub fn new(dt: DataLayout, LayoutAttrs { att }: LayoutAttrs) -> Result<Self, ErrorPosition> {
        if att.dt() != dt {
            return Err(locate_error!());
        }
        let &[nh, seq_len, att_len] = att.shape() else {
            return Err(locate_error!());
        };
        let &[stride_head, stride_token, stride_num] = att.strides() else {
            return Err(locate_error!());
        };
        let unit = dt.nbytes() as isize;
        if stride_num != unit {
            return Err(locate_error!());
        }
        Ok(Self {
            nh,
            seq_len,
            att_len,
            stride_head: stride_head / unit,
            stride_token: stride_token / unit,
            offset: att.offset(),
        })
    }
}
