use super::{AllReduce, Args};
use crate::{ByteOf, Hardware, Operator, QueueAlloc};
use std::marker::PhantomData;

/// A non-all-reduce operator that does nothing.
#[repr(transparent)]
pub struct NonAllReduce<H>(PhantomData<H>);

impl<H: Hardware> AllReduce<H, H> for NonAllReduce<H> {}

impl<H: Hardware> Operator for NonAllReduce<H> {
    type Hardware = H;
    type TopoNode = H;
    type Args = Args<H>;

    #[inline]
    fn new(_node: &Self::TopoNode) -> Self {
        Self(PhantomData)
    }

    #[inline]
    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, crate::SchemeError> {
        Ok(0)
    }

    #[inline]
    fn launch<QA>(
        &self,
        _args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), crate::LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        Ok(())
    }
}
