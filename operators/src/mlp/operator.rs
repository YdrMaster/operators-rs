use super::{args::Meta, Args, Mlp};
use crate::{
    dyn_, get_static, mat_mul, swiglu, utils::sizeof, ByteOf, Hardware, LaunchError, QueueAlloc,
    SchemeError, TensorLayout, Workspace, WorkspaceCollector,
};
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::marker::PhantomData;

pub struct Operator<Hardware, MatMul, Swiglu> {
    mat_mul: MatMul,
    swiglu: Swiglu,
    _phantom: PhantomData<Hardware>,
}

impl<H, M, S> Mlp<H> for Operator<H, M, S>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    S: swiglu::Swiglu<H>,
{
}

impl<H, M, S> crate::Operator for Operator<H, M, S>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    S: swiglu::Swiglu<H>,
{
    type Hardware = H;
    type Args = Args<H>;

    fn new(processor: &Self::Hardware) -> Self {
        Self {
            mat_mul: M::new(processor),
            swiglu: S::new(processor),
            _phantom: PhantomData,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        use std::ptr::{null, null_mut};

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
        } = args;

        // 如果不能保证 nt di 已知，用任意值初始化算子
        let (Some(&nt), Some(&di)) = (nt.get_static(), di.get_static()) else {
            let mut wc = WorkspaceCollector::new();

            let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
            wc.push_sub(self.mat_mul.scheme(
                &mat_mul::Args::new_null(layout.clone(), 1., layout.clone(), layout, 1.),
                max_workspace_size,
            )?);

            let layout = TensorLayout::new_dyn(dt, &[nt, di], &[dyn_(); 2]);
            wc.push_sub(self.swiglu.scheme(
                &swiglu::Args::new_layout(layout.clone(), layout),
                max_workspace_size,
            )?);

            return Ok(wc.cauculate(max_workspace_size));
        };

        let ele = sizeof(dt)?;
        let gate_up_layout = ArrayLayout::<3>::new_contiguous(&[nt, di * 2], BigEndian, ele);
        let gate_up_size = nt * di * 2 * ele;
        let workspace_size = max_workspace_size.saturating_sub(gate_up_size);

        let mut wc = WorkspaceCollector::new();
        wc.push_base(gate_up_size);

        wc.push_sub(self.mat_mul.scheme(
            &mat_mul::Args {
                c_layout: TensorLayout::new(dt, gate_up_layout.shape(), gate_up_layout.strides()),
                c_base: null_mut(),
                beta: 0.,
                a_layout: x_layout.clone(),
                a_base: *x_base,
                b_layout: w_gate_up_layout.clone(),
                b_base: *w_gate_up_base,
                alpha: 1.,
            },
            workspace_size,
        )?);

        let up_layout = gate_up_layout.tile_be(1, &[2, di]).index(1, 1);
        let swiglu_layout = TensorLayout::new(dt, up_layout.shape(), up_layout.strides());
        wc.push_sub(self.swiglu.scheme(
            &swiglu::Args {
                gate_layout: swiglu_layout.clone(),
                gate_base: null_mut(),
                up_layout: swiglu_layout.clone(),
                up_base: null_mut(),
            },
            workspace_size,
        )?);

        wc.push_sub(self.mat_mul.scheme(
            &mat_mul::Args {
                c_layout: y_layout.clone(),
                c_base: *y_base,
                beta: if *down_bias { 1. } else { 0. },
                a_layout: swiglu_layout,
                a_base: null(),
                b_layout: w_down_layout.clone(),
                b_base: *w_down_base,
                alpha: *down_alpha,
            },
            workspace_size,
        )?);

        Ok(wc.cauculate(max_workspace_size))
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
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
        } = args;

        let ele = sizeof(dt)?;
        get_static!(nt di);

        let gate_up_size = nt * di * 2 * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, gate_up_size);
        let (gate_up, workspace) = workspace.split_at_mut(gate_up_size);

        let gate_up_layout = ArrayLayout::<3>::new_contiguous(&[nt, di * 2], BigEndian, ele);
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: TensorLayout::new(dt, gate_up_layout.shape(), gate_up_layout.strides()),
                c_base: gate_up.as_mut_ptr(),
                beta: 0.,
                a_layout: x_layout.clone(),
                a_base: *x_base,
                b_layout: w_gate_up_layout.clone(),
                b_base: *w_gate_up_base,
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )?;

        let up_layout = gate_up_layout.tile_be(1, &[2, di]).index(1, 1);
        let swiglu_layout = TensorLayout::new(dt, up_layout.shape(), up_layout.strides());
        self.swiglu.launch(
            &swiglu::Args {
                gate_layout: swiglu_layout.clone(),
                gate_base: gate_up.as_mut_ptr(),
                up_layout: swiglu_layout.clone(),
                up_base: unsafe { gate_up.as_mut_ptr().byte_add(up_layout.offset()) },
            },
            workspace,
            queue_alloc,
        )?;

        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: y_layout.clone(),
                c_base: *y_base,
                beta: if *down_bias { 1. } else { 0. },
                a_layout: swiglu_layout,
                a_base: gate_up.as_ptr(),
                b_layout: w_down_layout.clone(),
                b_base: *w_down_base,
                alpha: *down_alpha,
            },
            workspace,
            queue_alloc,
        )?;

        Ok(())
    }
}
