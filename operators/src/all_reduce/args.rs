use super::ReduceOp;
use crate::{ConstPtr, Hardware, MutPtr, TensorLayout};

pub struct Args<H: Hardware> {
    pub dst_layout: TensorLayout,
    pub dst_base: MutPtr<H>,
    pub src_layout: TensorLayout,
    pub src_base: ConstPtr<H>,
    pub op: ReduceOp,
}
