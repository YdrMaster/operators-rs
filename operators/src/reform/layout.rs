use common::{locate_error, ErrorPosition, TensorLayout};

pub struct LayoutAttrs {
    pub dst: TensorLayout,
    pub src: TensorLayout,
}

pub(super) struct SchemeLayout {
    pub r: usize,
    pub c: usize,
    pub z: usize,
    pub dst_rs: isize,
    pub dst_cs: isize,
    pub dst_offset: usize,
    pub src_rs: isize,
    pub src_cs: isize,
    pub src_offset: usize,
}

impl SchemeLayout {
    pub fn new(LayoutAttrs { dst, src }: LayoutAttrs) -> Result<Self, ErrorPosition> {
        if dst.dt() != src.dt() {
            return Err(locate_error!("dst and src must have the same data type"));
        }
        if dst.shape() != src.shape() {
            return Err(locate_error!("dst and src must have the same shape"));
        }

        let unit = dst.dt().nbytes() as isize;
        let [r @ .., c, z] = dst.shape() else {
            return Err(locate_error!("dst and src must have at least 2 dimensions"));
        };
        let [dst_rs @ .., dst_cs, dst_zs] = dst.strides() else {
            unreachable!()
        };
        let [src_rs @ .., src_cs, src_zs] = src.strides() else {
            unreachable!()
        };
        assert_eq!(dst_rs.len(), r.len());
        assert_eq!(src_rs.len(), r.len());
        if *dst_zs != unit || *src_zs != unit {
            return Err(locate_error!());
        }

        let r = match r {
            [] => 1,
            &[r] => r,
            r => {
                for (i, r) in r.iter().enumerate().skip(1) {
                    let r = *r as isize;
                    if r * dst_rs[i] != dst_rs[i - 1] || r * src_rs[i] != src_rs[i - 1] {
                        return Err(locate_error!("目前要求前序维度必须连续"));
                    }
                }
                r.iter().product()
            }
        };
        let dst_rs = dst_rs.last().copied().unwrap_or(0);
        let src_rs = src_rs.last().copied().unwrap_or(0);

        Ok(Self {
            r,
            c: *c,
            z: z * unit as usize,
            dst_rs,
            dst_cs: *dst_cs,
            dst_offset: dst.offset(),
            src_rs,
            src_cs: *src_cs,
            src_offset: src.offset(),
        })
    }
}
