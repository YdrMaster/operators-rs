use super::{args::Meta, Args, Mlp};
use crate::{
    mat_mul, swiglu,
    utils::{get_static, sizeof},
};
use common::{
    dyn_, out_of_workspace, Argument, Handle, LaunchError, QueueOf, SchemeError, TensorLayout,
};
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::marker::PhantomData;

pub struct Operator<Handle, MatMul, Swiglu> {
    dt: Option<DigitLayout>,
    nt: Argument<usize>,
    di: Argument<usize>,

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
    fn workspace_size(&self) -> Option<usize> {
        let ele = self.dt?.nbytes()?;
        let nt = *self.nt.get_static()?;
        let di = *self.di.get_static()?;
        Some(nt * di * 2 * ele)
    }
}

impl<H, M, A> common::Operator for Operator<H, M, A>
where
    H: Handle,
    M: mat_mul::MatMul<H>,
    A: swiglu::Swiglu<H>,
{
    type Handle = H;
    type Args = Args<H>;

    #[inline]
    fn new(handle: &Self::Handle) -> Self {
        Self {
            dt: None,
            nt: dyn_(),
            di: dyn_(),

            mat_mul: M::new(handle),
            swiglu: A::new(handle),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), SchemeError> {
        use std::ptr::{null, null_mut};

        let Meta { dt, nt, di } = args.meta()?;

        self.dt = Some(dt);
        self.nt = nt;
        self.di = di;

        let layout = TensorLayout::new_dyn(dt, &[nt, di], &[dyn_(); 2]);
        self.swiglu.scheme(&swiglu::Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        })
    }

    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError> {
        let Meta { dt, nt, di } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            w_gate_up_layout,
            w_gate_up_base,
            w_down_layout,
            w_down_base,
            down_alpha,
            down_bias,
            workspace_size,
            workspace,
        } = args;

        get_static!(nt di);
        let ele = sizeof(dt)?;
        if *workspace_size < nt * di * 2 * ele {
            return Err(out_of_workspace(""));
        }

        let gate_up_layout = ArrayLayout::<3>::new_contiguous(&[nt, di * 2], BigEndian, ele);
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: TensorLayout::new(dt, gate_up_layout.shape(), gate_up_layout.strides()),
                c_base: *workspace,
                beta: 0.,
                a_layout: x_layout.clone(),
                a_base: *x_base,
                b_layout: w_gate_up_layout.clone(),
                b_base: *w_gate_up_base,
                alpha: 1.,
            },
            queue,
        )?;

        let up_layout = gate_up_layout.tile_be(1, &[2, di]).index(1, 1);
        let swiglu_layout = TensorLayout::new(dt, up_layout.shape(), up_layout.strides());
        self.swiglu.launch(
            &swiglu::Args {
                gate_layout: swiglu_layout.clone(),
                gate_base: *workspace,
                up_layout: swiglu_layout.clone(),
                up_base: unsafe { workspace.byte_add(up_layout.offset()) },
            },
            queue,
        )?;

        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: y_layout.clone(),
                c_base: *y_base,
                beta: if *down_bias { 1. } else { 0. },
                a_layout: swiglu_layout,
                a_base: *workspace,
                b_layout: w_down_layout.clone(),
                b_base: *w_down_base,
                alpha: *down_alpha,
            },
            queue,
        )
    }
}
