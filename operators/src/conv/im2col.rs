use super::{args::Meta, Args, Conv};
use crate::{
    args_not_support, get_static, mat_mul, rearrange, strides_not_support, ByteOf, Hardware,
    LaunchError, QueueAlloc, SchemeError, TensorLayout, Workspace,
};
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::marker::PhantomData;

pub struct Operator<Hardware, Rearrange, MatMul> {
    rearrange: Rearrange,
    mat_mul: MatMul,
    _phantom: PhantomData<Hardware>,
}

impl<H, R, M> Conv<H> for Operator<H, R, M>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    M: mat_mul::MatMul<H>,
{
}

impl<H, R, M> crate::Operator for Operator<H, R, M>
where
    H: Hardware,
    R: rearrange::Rearrange<H>,
    M: mat_mul::MatMul<H>,
{
    type Hardware = H;
    type TopoNode = H;
    type Args = Args<H>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            rearrange: R::new(node),
            mat_mul: M::new(node),
            _phantom: PhantomData,
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Args { pads, .. } = args;
        let &[0, 0, 0, 0] = pads else {
            return Err(args_not_support(
                "non-zero padding for im2col is not supported",
            ));
        };
        let Meta {
            dt,
            n,
            c,
            hy,
            wy,
            hk,
            wk,
            ..
        } = args.meta()?;

        macro_rules! get {
            ($( $var:ident )+) => {
                $(
                    let Some(&$var) = $var.get_static() else {
                        return Ok(0);
                    };
                )+
            };
        }

        get!(n c hk wk hy wy);
        let a_size = [n, c, hk, wk, hy, wy, dt.nbytes()].iter().product();

        Ok(if a_size <= max_workspace_size {
            a_size
        } else {
            0
        })
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
        let Args {
            y_layout,
            x_layout,
            w_layout,
            b_layout,
            strides,
            dilations,
            pads,

            y_base,
            x_base,
            w_base,
            b_base,
        } = args;
        let &[hs, ws] = strides;
        let &[hd, wd] = dilations;
        let &[0, 0, 0, 0] = pads else {
            return Err(args_not_support("non-zero padding for im2col is not supported").into());
        };

        let Meta {
            dt,
            n,
            m,
            c,
            h,
            w,
            hy,
            wy,
            hk,
            wk,
        } = args.meta()?;

        let &[nys, mys, hys, wys] = y_layout.strides() else {
            unreachable!()
        };
        let &[nxs, cxs, hxs, wxs] = x_layout.strides() else {
            unreachable!()
        };
        let &[mks, cks, hks, wks] = w_layout.strides() else {
            unreachable!()
        };
        let &[mbs] = b_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   m   c   h hy hk w wy wk
            nys mys     hys     wys
            nxs     cxs hxs     wxs
                mks cks hks     wks
                mbs
        }

        // 计算考虑空洞的 kernel size

        let hkd = (hk - 1) * hd + 1;
        let wkd = (wk - 1) * wd + 1;

        if (h - hkd) % hs != 0 || (w - wkd) % ws != 0 {
            return Err(strides_not_support("output size not divisible by strides").into());
        }

        type Arr6 = ArrayLayout<6>;

        // c <- y: [n, m, hy * wy]
        // a <- w: [n, m, c * hk * wk]
        // b <- x: [n, c * hk * wk, hy * wy]

        // y 作为矩阵乘输出的布局
        let Some(c_y) = Arr6::new(&[n, m, hy, wy], &[nys, mys, hys, wys], 0).merge(2..4) else {
            return Err(strides_not_support("").into());
        };
        // w 作为矩阵乘输入的布局
        let Some(b_w) = Arr6::new(&[n, m, c, hk, wk], &[0, mks, cks, hks, wks], 0).merge(2..5)
        else {
            return Err(strides_not_support("").into());
        };
        // x im2col rearrange
        let ele = dt.nbytes();
        let a_shape = [n, c, hk, wk, hy, wy];
        let [hd, wd, hs, ws] = [hd, wd, hs, ws].map(|x| x as isize);
        let a_strides = [nxs, cxs, hxs * hd, wxs * wd, hxs * hs, wxs * ws];
        let a_dst = Arr6::new_contiguous(&a_shape, BigEndian, ele);
        let a_src = Arr6::new(&a_shape, &a_strides, 0);
        let a_x = a_dst.merge_many(&[1..4, 4..6]).unwrap();

        let c_y = TensorLayout::from_arr(dt, &c_y);
        let b_w = TensorLayout::from_arr(dt, &b_w);
        let a_x = TensorLayout::from_arr(dt, &a_x);
        let a_dst = TensorLayout::from_arr(dt, &a_dst);
        let a_src = TensorLayout::from_arr(dt, &a_src);

        // b 布局广播
        let b = Arr6::new(&[n, m, hy * wy], &[0, mbs, 0], 0);
        // 广播 b
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: c_y.clone(),
                dst_base: *y_base,
                src_layout: TensorLayout::new(dt, b.shape(), b.strides()),
                src_base: *b_base,
            },
            workspace,
            queue_alloc,
        )?;

        // 为 im2col 分配工作空间
        let a_size = a_shape.iter().product::<usize>() * ele;
        let mut workspace = Workspace::new(queue_alloc, workspace, a_size);
        let (a_mem, workspace) = workspace.split_at_mut(a_size);
        // im2col 变换
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: a_dst,
                dst_base: a_mem.as_mut_ptr(),
                src_layout: a_src,
                src_base: *x_base,
            },
            workspace,
            queue_alloc,
        )?;

        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: c_y.clone(),
                c_base: *y_base,
                beta: 1.,
                a_layout: a_x,
                a_base: a_mem.as_ptr(),
                b_layout: b_w,
                b_base: *w_base,
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )
    }
}
