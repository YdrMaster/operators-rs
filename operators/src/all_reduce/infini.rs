use super::{args::Meta, AllReduce, Args, ReduceOp};
use crate::{
    infini::{Device, InfiniNode},
    rearrange::{self, infini::Operator as Rearrange},
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::types as ty;
use infini_ccl::bindings::InfiniDataType_t;
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub enum Operator {
    Rearrange(Rearrange),
    Comm(Arc<infini_ccl::Comm>),
}

impl AllReduce<Device, InfiniNode> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = InfiniNode;
    type Args = Args<Device>;

    fn new(node: &Self::TopoNode) -> Self {
        match node.comm.as_ref() {
            Some(comm) => Self::Comm(comm.clone()),
            None => Self::Rearrange(Rearrange::new(&node.device)),
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        match self {
            Self::Rearrange(rearrange) => rearrange.launch(&args.pair, workspace, queue_alloc),
            Self::Comm(comm) => {
                let Meta { dt, size } = args.meta()?;
                let &Args {
                    pair:
                        rearrange::Args {
                            dst_base, src_base, ..
                        },
                    op,
                    ..
                } = args;

                assert_eq!(op, ReduceOp::Sum);
                let len = dt.nbytes() * size;

                comm.allreduce_sum(
                    unsafe { from_raw_parts_mut(dst_base, len) },
                    unsafe { from_raw_parts(src_base, len) },
                    match dt {
                        ty::F16 => InfiniDataType_t::INFINI_F16,
                        ty::F32 => InfiniDataType_t::INFINI_F32,
                        _ => todo!(),
                    },
                    queue_alloc.queue(),
                );
                Ok(())
            }
        }
    }
}
