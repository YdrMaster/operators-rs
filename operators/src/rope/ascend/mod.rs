use super::{Args, Rope, Seq, SinCosTable};
use crate::{ascend::Npu, ByteOf, LaunchError, QueueAlloc, SchemeError};
use digit_layout::DigitLayout;

pub struct Operator;

impl Rope<Npu> for Operator {
    fn build_sincos<QA>(
        _dt: DigitLayout,
        _nctx: usize,
        _dh: usize,
        _queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        todo!()
    }

    fn build_pos<I, QA>(
        _dt: digit_layout::DigitLayout,
        _nt: usize,
        _iter: I,
        _queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        todo!()
    }
}

impl crate::Operator for Operator {
    type Hardware = Npu;
    type TopoNode = Npu;
    type Args = Args<Npu>;

    fn new(_node: &Self::TopoNode) -> Self {
        todo!()
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        todo!()
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
        todo!()
    }
}
