use super::{args::Meta, Args, Gpt2Mlp};
use crate::{
    dyn_, gelu, get_static, mat_mul, rearrange, ByteOf, Hardware, LaunchError, QueueAlloc,
    SchemeError, TensorLayout, Workspace, WorkspaceCollector,
};
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::marker::PhantomData;

pub struct Operator<Hardware, MatMul, Gelu, Rearrange> {
    mat_mul: MatMul,
    gelu: Gelu,
    rearrange: Rearrange,
    _phantom: PhantomData<Hardware>,
}

impl<H, M, G, R> Gpt2Mlp<H> for Operator<H, M, G, R>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    G: gelu::Gelu<H>,
    R: rearrange::Rearrange<H>,
{
}

impl<H, M, G, R> crate::Operator for Operator<H, M, G, R>
where
    H: Hardware,
    M: mat_mul::MatMul<H>,
    G: gelu::Gelu<H>,
    R: rearrange::Rearrange<H>,
{
    type Hardware = H;
    type TopoNode = H;
    type Args = Args<H>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            mat_mul: M::new(node),
            gelu: G::new(node),
            rearrange: R::new(node),
            _phantom: PhantomData,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        use std::ptr::{null, null_mut};

        let Meta { dt, nt, di, d } = args.meta()?;

        let Args {
            y_layout,
            y_base,
            up_weight_layout,
            up_weight_base,
            up_bias_base,
            down_weight_layout,
            down_weight_base,
            down_bias_base,
            ..
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
            wc.push_sub(
                self.gelu
                    .scheme(&gelu::Args::new_layout(layout), max_workspace_size)?,
            );
            let layout = TensorLayout::new_dyn(dt, &[nt, di], &[dyn_(); 2]);
            wc.push_sub(self.rearrange.scheme(
                &rearrange::Args::new_null(layout.clone(), layout.clone()),
                max_workspace_size,
            )?);
            return Ok(wc.cauculate(max_workspace_size));
        };

        let ele = dt.nbytes();
        let tmp_size = nt * di * ele;
        let tmp_layout = ArrayLayout::<3>::new_contiguous(&[nt, di], BigEndian, ele);
        let tmp_layout = TensorLayout::new(dt, tmp_layout.shape(), tmp_layout.strides());
        let workspace_size = max_workspace_size.saturating_sub(tmp_size);

        let mut wc = WorkspaceCollector::new();
        wc.push_base(tmp_size);
        wc.push_base(workspace_size);
        // Conv1D
        {
            let up_bias_layout =
                ArrayLayout::<3>::new_contiguous(&[1, di], BigEndian, ele).broadcast(0, nt);
            wc.push_sub(self.rearrange.scheme(
                &rearrange::Args {
                    dst_layout: tmp_layout.clone(),
                    dst_base: null_mut(),
                    src_layout: TensorLayout::new(
                        dt,
                        up_bias_layout.shape(),
                        up_bias_layout.strides(),
                    ),
                    src_base: *up_bias_base,
                },
                workspace_size,
            )?);
            wc.push_sub(self.mat_mul.scheme(
                &mat_mul::Args {
                    c_layout: tmp_layout.clone(),
                    c_base: null_mut(),
                    beta: 1.,
                    a_layout: y_layout.clone(),
                    a_base: *y_base,
                    b_layout: up_weight_layout.clone(),
                    b_base: *up_weight_base,
                    alpha: 1.,
                },
                workspace_size,
            )?);
        }
        // gelu
        wc.push_sub(self.gelu.scheme(
            &gelu::Args {
                layout: tmp_layout.clone(),
                base: null_mut(),
            },
            workspace_size,
        )?);
        // Conv1D
        {
            let down_bias_layout =
                ArrayLayout::<3>::new_contiguous(&[1, *d.get_static().unwrap()], BigEndian, ele)
                    .broadcast(0, nt);
            wc.push_sub(self.rearrange.scheme(
                &rearrange::Args {
                    dst_layout: y_layout.clone(),
                    dst_base: *y_base,
                    src_layout: TensorLayout::new(
                        dt,
                        down_bias_layout.shape(),
                        down_bias_layout.strides(),
                    ),
                    src_base: *down_bias_base,
                },
                workspace_size,
            )?);
            wc.push_sub(self.mat_mul.scheme(
                &mat_mul::Args {
                    c_layout: y_layout.clone(),
                    c_base: *y_base,
                    beta: 1.,
                    a_layout: tmp_layout.clone(),
                    a_base: null(),
                    b_layout: down_weight_layout.clone(),
                    b_base: *down_weight_base,
                    alpha: 1.,
                },
                workspace_size,
            )?);
        }

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
        let Meta { dt, nt, di, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            up_weight_layout,
            up_weight_base,
            up_bias_base,
            down_weight_layout,
            down_weight_base,
            down_bias_base,
            ..
        } = args;

        let ele = dt.nbytes();
        get_static!(nt di d);

        let tmp_size = nt * di * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, tmp_size);
        let (tmp, workspace) = workspace.split_at_mut(tmp_size);

        let tmp_layout = ArrayLayout::<3>::new_contiguous(&[nt, di], BigEndian, ele);
        let tmp_layout = TensorLayout::new(dt, tmp_layout.shape(), tmp_layout.strides());
        // Conv1D
        {
            let up_bias_layout =
                ArrayLayout::<3>::new_contiguous(&[1, di], BigEndian, ele).broadcast(0, nt);
            self.rearrange.launch(
                &rearrange::Args {
                    dst_layout: tmp_layout.clone(),
                    dst_base: tmp.as_mut_ptr(),
                    src_layout: TensorLayout::new(
                        dt,
                        up_bias_layout.shape(),
                        up_bias_layout.strides(),
                    ),
                    src_base: *up_bias_base,
                },
                workspace,
                queue_alloc,
            )?;

            self.mat_mul.launch(
                &mat_mul::Args {
                    c_layout: tmp_layout.clone(),
                    c_base: tmp.as_mut_ptr(),
                    beta: 1.,
                    a_layout: y_layout.clone(),
                    a_base: *y_base,
                    b_layout: up_weight_layout.clone(),
                    b_base: *up_weight_base,
                    alpha: 1.,
                },
                workspace,
                queue_alloc,
            )?;
        }
        // gelu
        self.gelu.launch(
            &gelu::Args {
                layout: tmp_layout.clone(),
                base: tmp.as_mut_ptr(),
            },
            workspace,
            queue_alloc,
        )?;
        // Conv1D
        {
            let down_bias_layout =
                ArrayLayout::<3>::new_contiguous(&[1, d], BigEndian, ele).broadcast(0, nt);
            self.rearrange.launch(
                &rearrange::Args {
                    dst_layout: y_layout.clone(),
                    dst_base: *y_base,
                    src_layout: TensorLayout::new(
                        dt,
                        down_bias_layout.shape(),
                        down_bias_layout.strides(),
                    ),
                    src_base: *down_bias_base,
                },
                workspace,
                queue_alloc,
            )?;

            self.mat_mul.launch(
                &mat_mul::Args {
                    c_layout: y_layout.clone(),
                    c_base: *y_base,
                    beta: 1.,
                    a_layout: tmp_layout.clone(),
                    a_base: tmp.as_ptr(),
                    b_layout: down_weight_layout.clone(),
                    b_base: *down_weight_base,
                    alpha: 1.,
                },
                workspace,
                queue_alloc,
            )?;
        }

        Ok(())
    }
}
