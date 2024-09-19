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

    pub k_cache_layout: TensorLayout,
    pub k_cache_base: MutPtr<H>,

    pub v_cache_layout: TensorLayout,
    pub v_cache_base: MutPtr<H>,

    pub pos: usize,

    pub workspace: Workspace<H>,
}

pub(super) struct Meta {
    pub dt: DigitLayout,
    pub nh: Argument<usize>,
    pub nkvh: Argument<usize>,
    pub dh: Argument<usize>,

    pub seq: Argument<usize>,
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        let dt = self.q_layout.dt();
        pass_if! {
            self.      k_layout.dt() == dt;
            self.      v_layout.dt() == dt;
            self.      o_layout.dt() == dt;
            self.k_cache_layout.dt() == dt;
            self.v_cache_layout.dt() == dt;
        }
        pass_match! {
            &[nh_q   , seq_q , dh_q ] = self.      q_layout.shape();
            &[nkvh_k , seq_k , dh_k ] = self.      k_layout.shape();
            &[nkvh_v , seq_v , dh_v ] = self.      v_layout.shape();
            &[nh_o   , seq_o , dh_o ] = self.      o_layout.shape();
            &[nkvh_kc,  _buf , dh_kc] = self.k_cache_layout.shape();
            &[nkvh_vc,  _buf , dh_vc] = self.v_cache_layout.shape();

            Ok(&nh  ) = Argument::merge(&[nh_q, nh_o]);
            Ok(&nkvh) = Argument::merge(&[nkvh_k, nkvh_v, nkvh_kc, nkvh_vc]);
            Ok(&dh  ) = Argument::merge(&[dh_q, dh_k, dh_v, dh_o, dh_kc, dh_vc]);
            Ok(&seq ) = Argument::merge(&[seq_q, seq_k, seq_v, seq_o]);

        }
        Ok(Meta {
            dt,
            nh,
            nkvh,
            dh,
            seq,
        })
    }
}
