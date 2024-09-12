use super::{args::Meta, Args, Attention};
use crate::{fuesd_softmax, mat_mul, rearrange};
use common::{dyn_, ErrorPosition, Handle, QueueOf, TensorLayout};
use std::{marker::PhantomData, ptr::null_mut};

#[allow(dead_code)]
pub struct Operator<Handle, MatMul, Softmax, Rearrange> {
    mat_mul: MatMul,
    softmax: Softmax,
    rearrange: Rearrange,
    _phantom: PhantomData<Handle>,
}

impl<H, M, S, R> Attention<H> for Operator<H, M, S, R>
where
    H: Handle,
    M: mat_mul::MatMul<H>,
    S: fuesd_softmax::FusedSoftmax<H>,
    R: rearrange::Rearrange<H>,
{
}

impl<H, M, S, R> common::Operator for Operator<H, M, S, R>
where
    H: Handle,
    M: mat_mul::MatMul<H>,
    S: fuesd_softmax::FusedSoftmax<H>,
    R: rearrange::Rearrange<H>,
{
    type Handle = H;
    type Args = Args<H>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            mat_mul: M::new(handle),
            softmax: S::new(handle),
            rearrange: R::new(handle),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let Meta {
            dt, nh, seq, att, ..
        } = args.meta()?;
        self.softmax.scheme(&fuesd_softmax::Args {
            att_layout: TensorLayout::new(dt, &[nh, seq, att], &[dyn_(); 3]),
            att_base: null_mut(),
        })
    }

    fn launch(
        &self,
        args: &Self::Args,
        _queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        #[allow(unused_variables)]
        let Meta {
            dt,
            nh,
            nkvh,
            seq,
            att,
            dh,
        } = args.meta()?;

        todo!()
    }
}
