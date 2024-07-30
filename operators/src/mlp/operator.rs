use super::{Args, Mlp};
use crate::{mat_mul, swiglu};
use common::{ErrorPosition, Handle, QueueOf};
use std::marker::PhantomData;

pub struct Operator<Handle, MatMul, Swiglu> {
    mat_mul: MatMul,
    swiglu: Swiglu,
    _phantom: PhantomData<Handle>,
}

impl<H, M, A> Mlp<H> for Operator<H, M, A>
where
    H: Handle,
    M: mat_mul::MatMul<H>,
    A: swiglu::Swiglu<H>,
{
}

impl<H, M, A> common::Operator for Operator<H, M, A>
where
    H: Handle,
    M: mat_mul::MatMul<H>,
    A: swiglu::Swiglu<H>,
{
    type Handle = H;
    type Args = Args<H>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            mat_mul: M::new(handle),
            swiglu: A::new(handle),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        self.swiglu.scheme(&args.swiglu_args())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        self.mat_mul.launch(&args.gate_up_args(), queue)?;
        self.swiglu.launch(&args.swiglu_args(), queue)?;
        self.mat_mul.launch(&args.down_args(), queue)
    }
}
