use crate::utils::{dim_distinct, rank_not_support, type_distinct, ConstPtr, MutPtr};
use common::{Argument, Handle, ParamError, TensorLayout};
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

    pub workspace_size: usize,
    pub workspace: MutPtr<H>,
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
            workspace_size: usize::MAX,
            workspace: null_mut(),
        }
    }
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ParamError> {
        let Self {
            q_layout,
            k_layout,
            v_layout,
            o_layout,
            ..
        } = self;

        let &[nh_q, seq_q, dh_q] = self.q_layout.shape() else {
            return Err(rank_not_support("q", 3, q_layout.ndim()));
        };
        let &[nkvh_k, att_k, dh_k] = self.k_layout.shape() else {
            return Err(rank_not_support("k", 3, k_layout.ndim()));
        };
        let &[nkvh_v, att_v, dh_v] = self.v_layout.shape() else {
            return Err(rank_not_support("v", 3, v_layout.ndim()));
        };
        let &[nh_o, seq_o, dh_o] = self.o_layout.shape() else {
            return Err(rank_not_support("o", 3, o_layout.ndim()));
        };

        Ok(Meta {
            dt: type_distinct(&[q_layout.dt(), k_layout.dt(), v_layout.dt(), o_layout.dt()])?,
            nh: dim_distinct(&[nh_q, nh_o])?,
            nkvh: dim_distinct(&[nkvh_k, nkvh_v])?,
            seq: dim_distinct(&[seq_q, seq_o])?,
            att: dim_distinct(&[att_k, att_v])?,
            dh: dim_distinct(&[dh_q, dh_k, dh_v, dh_o])?,
        })
    }
}
