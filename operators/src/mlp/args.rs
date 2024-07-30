use crate::{
    mat_mul, swiglu,
    utils::{ConstPtr, MutPtr},
};
use common::{Handle, TensorLayout};
use std::ptr::null;

pub struct Args<H: Handle> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,

    pub x_layout: TensorLayout,
    pub x_base: ConstPtr<H>,

    pub gate_up_layout: TensorLayout,
    pub gate_up_base: MutPtr<H>,

    pub w_gate_up_layout: TensorLayout,
    pub w_gate_up_base: ConstPtr<H>,

    pub w_down_layout: TensorLayout,
    pub w_down_base: ConstPtr<H>,
    pub down_alpha: f32,
    pub down_bias: bool,
}

impl<H: Handle> Args<H> {
    pub(super) fn gate_up_args(&self) -> mat_mul::Args<H> {
        mat_mul::Args {
            c_layout: self.gate_up_layout.clone(),
            c_base: self.gate_up_base,
            beta: 0.,
            a_layout: self.x_layout.clone(),
            a_base: self.x_base,
            b_layout: self.w_gate_up_layout.clone(),
            b_base: self.w_gate_up_base,
            alpha: 1.,
        }
    }

    pub(super) fn swiglu_args(&self) -> swiglu::Args<H> {
        let layout = self.gate_up_layout();
        let up_base = if self.gate_up_base.is_null() {
            null()
        } else {
            let d = *layout.shape()[1].get_static().unwrap() as isize;
            let s = *layout.strides()[1].get_static().unwrap();
            unsafe { self.gate_up_base.offset(d * s) }
        };
        swiglu::Args {
            gate_layout: layout.clone(),
            gate_base: self.gate_up_base,
            up_layout: layout,
            up_base,
        }
    }

    pub(super) fn down_args(&self) -> mat_mul::Args<H> {
        mat_mul::Args {
            c_layout: self.y_layout.clone(),
            c_base: self.y_base,
            beta: if self.down_bias { 1. } else { 0. },
            a_layout: self.gate_up_layout(),
            a_base: self.gate_up_base,
            b_layout: self.w_down_layout.clone(),
            b_base: self.w_down_base,
            alpha: self.down_alpha,
        }
    }

    fn gate_up_layout(&self) -> TensorLayout {
        let &[n, d] = self.gate_up_layout.shape() else {
            unreachable!()
        };
        TensorLayout::new(
            self.gate_up_layout.dt(),
            [
                n,
                if let Some(d) = d.get_static() {
                    (d / 2).into()
                } else {
                    d
                },
            ],
            self.gate_up_layout.strides(),
        )
    }
}
