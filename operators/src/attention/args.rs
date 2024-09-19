use crate::utils::{pass_if, pass_match, ConstPtr, MutPtr};
use common::{Argument, ErrorPosition, Handle, TensorLayout, Workspace};
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

pub(crate) struct Meta {
    pub dt: DigitLayout,
    pub nh: Argument<usize>,
    pub nkvh: Argument<usize>,
    pub seq: Argument<usize>,
    pub att: Argument<usize>,
    pub dh: Argument<usize>,
}

impl<H: Handle> From<Meta> for Args<H> {
    fn from(value: Meta) -> Self {
        use common::dyn_;
        use std::ptr::{null, null_mut};

        let Meta {
            dt,
            nh,
            nkvh,
            seq,
            att,
            dh,
        } = value;

        let qo_layout = TensorLayout::new_dyn(dt, &[nh, seq, dh], &[dyn_(); 3]);
        let kv_layout = TensorLayout::new_dyn(dt, &[nkvh, att, dh], &[dyn_(); 3]);
        Self {
            q_layout: qo_layout.clone(),
            q_base: null_mut(),
            k_layout: kv_layout.clone(),
            k_base: null(),
            v_layout: kv_layout,
            v_base: null(),
            o_layout: qo_layout,
            o_base: null_mut(),
            workspace: Workspace {
                ptr: null_mut(),
                len: 0,
            },
        }
    }
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.q_layout.dt();
        pass_if! {
            self.k_layout.dt() == dt;
            self.v_layout.dt() == dt;
            self.o_layout.dt() == dt;
        }
        pass_match! {
            &[nh_q  , seq_q, dh_q] = self.q_layout.shape();
            &[nkvh_k, att_k, dh_k] = self.k_layout.shape();
            &[nkvh_v, att_v, dh_v] = self.v_layout.shape();
            &[nh_o  , seq_o, dh_o] = self.o_layout.shape();
            Ok(&nh  ) = Argument::merge(&[nh_q  , nh_o  ]);
            Ok(&nkvh) = Argument::merge(&[nkvh_k, nkvh_v]);
            Ok(&seq ) = Argument::merge(&[seq_q , seq_o ]);
            Ok(&att ) = Argument::merge(&[att_k , att_v ]);
            Ok(&dh  ) = Argument::merge(&[dh_q, dh_k, dh_v, dh_o]);
        }
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
