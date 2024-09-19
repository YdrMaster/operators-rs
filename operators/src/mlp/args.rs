use crate::utils::{ConstPtr, MutPtr};
use common::{dyn_, pass_if, pass_match, Argument, ErrorPosition, Handle, TensorLayout, Workspace};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub y_layout: TensorLayout,
    pub y_base: MutPtr<H>,

    pub x_layout: TensorLayout,
    pub x_base: ConstPtr<H>,

    pub w_gate_up_layout: TensorLayout,
    pub w_gate_up_base: ConstPtr<H>,

    pub w_down_layout: TensorLayout,
    pub w_down_base: ConstPtr<H>,
    pub down_alpha: f32,
    pub down_bias: bool,

    pub workspace: Workspace<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nt: Argument<usize>,
    pub di: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.y_layout.dt();
        pass_if! {
            self.        x_layout.dt() == dt;
            self.w_gate_up_layout.dt() == dt;
            self.   w_down_layout.dt() == dt;
        }
        pass_match! {
            &[nt_y, d_y  ] = self.y_layout.shape();
            &[nt_x, d_x  ] = self.x_layout.shape();
            &[d_gu, di_gu] = self.w_gate_up_layout.shape();
            &[di_d, d_d  ] = self.w_down_layout.shape();
            Ok(&nt) = Argument::merge(&[nt_y, nt_x]);
            Ok(&_ ) = Argument::merge(&[d_y, d_x, d_gu, d_d]);
        }
        let di = if let Some(&di2) = di_gu.get_static() {
            pass_match!(Ok(&di) = Argument::merge(&[di_d, (di2 / 2).into()]));
            di
        } else {
            dyn_()
        };
        Ok(Meta { dt, nt, di })
    }
}
