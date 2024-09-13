use crate::utils::{ConstPtr, MutPtr};
use common::{pass_if, pass_match, Argument, ErrorPosition, Handle, TensorLayout, Workspace};
use digit_layout::DigitLayout;

pub struct Args<H: Handle> {
    pub q_layout: TensorLayout,
    pub q_base: MutPtr<H>,

    pub k_layout: TensorLayout,
    pub k_base: ConstPtr<H>,

    pub v_layout: TensorLayout,
    pub v_base: ConstPtr<H>,

    pub o_layout: TensorLayout,
    pub o_base: MutPtr<H>,

    pub workspace: Workspace<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: Argument<usize>,
    pub nkvh: Argument<usize>,
    pub seq: Argument<usize>,
    pub att: Argument<usize>,
    pub dh: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.q_layout.dt();

        pass_if!(self.k_layout.dt() == dt && self.v_layout.dt() == dt && self.o_layout.dt() == dt);
        pass_match!(&[nh_q, seq_q, dh_q] = self.q_layout.shape());
        pass_match!(&[nkvh_k, att_k, dh_k] = self.k_layout.shape());
        pass_match!(&[nkvh_v, att_v, dh_v] = self.v_layout.shape());
        pass_match!(&[nh_o, seq_o, dh_o] = self.o_layout.shape());
        pass_match!(Ok(&nh) = Argument::merge(&[nh_q, nh_o]));
        pass_match!(Ok(&nkvh) = Argument::merge(&[nkvh_k, nkvh_v]));
        pass_match!(Ok(&seq) = Argument::merge(&[seq_q, seq_o]));
        pass_match!(Ok(&att) = Argument::merge(&[att_k, att_v]));
        pass_match!(Ok(&dh) = Argument::merge(&[dh_q, dh_k, dh_v, dh_o]));
        Ok(Meta {
            dt,
            nh,
            nkvh,
            seq,
            att,
            dh,
        })
    }
}
