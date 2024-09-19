use super::{args::Meta, Args, Attention};
use crate::{fuesd_softmax, mat_mul, rearrange, utils::get_or_err};
use common::{
    algebraic, dyn_, locate_error, pass_match, Argument, ErrorPosition, Handle, QueueOf,
    TensorLayout,
};
use digit_layout::DigitLayout;
use ndarray_layout::ArrayLayout;
use std::marker::PhantomData;

pub struct Operator<Handle, MatMul, Softmax, Rearrange> {
    dt: Option<DigitLayout>,
    nh: Argument<usize>,
    seq: Argument<usize>,
    att: Argument<usize>,

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
    fn workspace_size(&self) -> Option<usize> {
        let ele = self.dt?.nbytes()?;
        let nh = *self.nh.get_static()?;
        let seq = *self.seq.get_static()?;
        let att = *self.att.get_static()?;
        Some(nh * seq * att * ele)
    }
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
            dt: None,
            nh: dyn_(),
            seq: dyn_(),
            att: dyn_(),

            mat_mul: M::new(handle),
            softmax: S::new(handle),
            rearrange: R::new(handle),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        use std::ptr::{null, null_mut};

        let Meta {
            dt,
            nh,
            seq,
            att,
            dh,
            ..
        } = args.meta()?;

        self.dt = Some(dt);
        self.nh = nh;
        self.seq = seq;
        self.att = att;

        self.softmax.scheme(&fuesd_softmax::Args {
            att_layout: TensorLayout::new_dyn(dt, &[nh, seq, att], &[dyn_(); 3]),
            att_base: null_mut(),
        })?;

        let rearrange_layout = TensorLayout::new_dyn(dt, &[nh, seq, dh], &[dyn_(); 3]);
        self.rearrange.scheme(&rearrange::Args {
            dst_layout: rearrange_layout.clone(),
            dst_base: null_mut(),
            src_layout: rearrange_layout,
            src_base: null(),
        })
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta {
            dt,
            nh,
            nkvh,
            seq,
            att,
            dh,
        } = args.meta()?;
        let Args {
            q_layout,
            q_base,
            k_layout,
            k_base,
            v_layout,
            v_base,
            o_layout,
            o_base,
            workspace,
        } = args;

        pass_match! {
            &[nh_sq  , seq_sq, dh_sq ] = q_layout.strides();
            &[nkvh_sk, seq_sk, att_sk] = k_layout.strides();
        }
        get_or_err! {
            nh      seq    dh
            nh_sq   seq_sq dh_sq
            nkvh           att
            nkvh_sk seq_sk att_sk
        };
        if workspace.len < nh * seq * att * algebraic!(dt)? {
            return Err(locate_error!("Out of workspace"));
        }

        #[inline(always)]
        fn layout(shape: [usize; 3], strides: [isize; 3]) -> ArrayLayout<4> {
            ArrayLayout::new(&shape, &strides, 0)
        }
        let q_layout = layout([nh, seq, dh], [nh_sq, seq_sq, dh_sq]);
        let k_layout = layout([nkvh, seq, att], [nkvh_sk, seq_sk, att_sk]);

        let head_group = nh / nkvh;
        let q_layout = q_layout.tile_be(0, &[nkvh, head_group]).merge(1..3);
        let k_layout = k_layout.transpose(&[2, 1]);

        assert_eq!(q_layout.offset(), 0);
        assert_eq!(k_layout.offset(), 0);
        let q_layout = TensorLayout::new(dt, q_layout.shape(), q_layout.strides());
        let k_layout = TensorLayout::new(dt, k_layout.shape(), k_layout.strides());
        let att_mat_mul = TensorLayout::new_contiguous(dt, &[nkvh, head_group * seq, att]);
        let att_softmax = TensorLayout::new_contiguous(dt, &[nh, seq, att]);

        // att = q . k^T
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: att_mat_mul.clone(),
                c_base: workspace.ptr,
                beta: 0.,
                a_layout: q_layout.clone(),
                a_base: *q_base,
                b_layout: k_layout,
                b_base: *k_base,
                alpha: (dh as f32).sqrt().recip(),
            },
            queue,
        )?;
        // att = softmax(att)
        self.softmax.launch(
            &fuesd_softmax::Args {
                att_layout: att_softmax,
                att_base: workspace.ptr,
            },
            queue,
        )?;
        // q = att . v
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: q_layout.clone(),
                c_base: *q_base,
                beta: 0.,
                a_layout: att_mat_mul,
                a_base: workspace.ptr,
                b_layout: v_layout.clone(),
                b_base: *v_base,
                alpha: 1.,
            },
            queue,
        )?;
        // o = rearrange(q)
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: o_layout.clone(),
                dst_base: *o_base,
                src_layout: q_layout,
                src_base: *q_base,
            },
            queue,
        )
    }
}
