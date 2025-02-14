use super::{args::Scheme, Args, Scale};
use crate::{
    cuda::{dt_name, Gpu, Handle, ModuleBox},
    shape_not_support, strides_not_support,
    utils::{gcd, type_distinct},
    ByteOf, LaunchError, QueueAlloc, SchemeDiversity, SchemeError,
};
use digit_layout::DigitLayout;
use lru::LruCache;
use std::{
    ffi::{c_uint, CString},
    sync::{Arc, Mutex},
};

pub struct Operator {}
impl Scale<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {}
    }

    #[inline]
    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        todo!();
        Ok(0)
    }

    fn launch<QA>(
        &self,
        _args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        todo!();
        Ok(())
    }
}
