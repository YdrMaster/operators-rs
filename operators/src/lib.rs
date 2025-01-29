// #![deny(warnings)]

mod common;
mod handle;

pub mod add;
pub mod add_rows;
pub mod all_reduce;
pub mod attention;
pub mod attention_kv_cached;
pub mod broadcast;
pub mod conv;
pub mod fuesd_softmax;
pub mod gelu;
pub mod layer_norm;
pub mod mat_mul;
pub mod random_sample;
pub mod rearrange;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;

pub use common::*;

#[cfg(any(use_cpu, test))]
pub use handle::common_cpu;

#[cfg(use_cl)]
pub use handle::opencl;
#[cfg(use_cl)]
pub extern crate clrt;

#[cfg(use_infini)]
pub use handle::infini;
#[cfg(use_infini)]
pub extern crate infini_rt;

#[cfg(use_cuda)]
pub mod cuda {
    pub use crate::handle::cuda::*;
    pub use ::cuda::*;
}
#[cfg(use_cuda)]
pub extern crate cublas;
#[cfg(use_nccl)]
pub extern crate nccl;

use rearrange::Rearrange;
use std::{marker::PhantomData, ops::DerefMut, ptr::addr_eq};

/// 算力硬件抽象。
///
/// 约定硬件如何存储和运行。
/// 这个特质应该由管理硬件的基本单元的映射类型实现，通常是**硬件上下文**。
pub trait Hardware {
    /// 硬件的存储单元类型。
    type Byte;
    /// 硬件的任务队列类型。
    type Queue<'ctx>;
}

pub trait TopoNode<H> {
    fn processor(&self) -> &H;
    fn rank(&self) -> usize;
    fn group_size(&self) -> usize;
}

impl<H: Hardware> TopoNode<H> for H {
    #[inline]
    fn processor(&self) -> &H {
        self
    }
    #[inline]
    fn rank(&self) -> usize {
        0
    }
    #[inline]
    fn group_size(&self) -> usize {
        1
    }
}

pub type ByteOf<H> = <H as Hardware>::Byte;
pub type QueueOf<'ctx, H> = <H as Hardware>::Queue<'ctx>;
pub type ArgsOf<O> = <O as Operator>::Args;
pub(crate) type MutPtr<H> = *mut <H as Hardware>::Byte;
pub(crate) type ConstPtr<H> = *const <H as Hardware>::Byte;

pub trait Alloc<M> {
    fn alloc(&self, size: usize) -> M;
    fn free(&self, mem: M);
}

/// 绑定到队列的分配器。
pub trait QueueAlloc: Alloc<Self::DevMem> {
    /// 队列分配器对应的硬件。
    type Hardware: Hardware;
    /// 分配器分配和回收的对象，表示对某块存储区域的所有权。
    type DevMem: DerefMut<Target = [ByteOf<Self::Hardware>]>;
    /// 分配器对应的队列。
    fn queue(&self) -> &QueueOf<Self::Hardware>;
}

/// 算子。
pub trait Operator {
    /// 执行算子的硬件。
    type Hardware: Hardware;
    /// 算子对应的通信拓扑节点。
    type TopoNode: TopoNode<Self::Hardware>;
    /// 算子的参数类型。
    type Args;

    /// 在指定拓扑节点上创建算子实例。
    fn new(node: &Self::TopoNode) -> Self;

    /// 规划执行方案。
    ///
    /// 通过向算子实例提供尽可能详细的参数来尽量确定算子执行方案。
    /// 通过允许参数中标量值、张量形状、张量步长和张量基址的动态性（[ArgVal] 或 [null](std::ptr::null)）来尽可能复用算子实例。
    ///
    /// 另外，需要传入一个最大工作空间容量。工作空间是与硬件存储单元相同类型的存储区域，供算子执行过程中使用。
    /// 规划执行方案时，将尽可能尝试计算一个满足最大工作空间容量的工作空间需求，作为返回值。
    ///
    /// 算子的返回值将保证不大于最大工作空间容量。如果算子还需要更多空间，可能产生运行时分配。
    ///
    /// 由于参数提供可能不全，有时无法计算出具体的工作空间需求，算子将返回 0 作为工作空间需求，并在执行时再计算实际的需求。
    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, SchemeError>;

    /// 发射算子到任务队列。
    ///
    /// 如果算子实际需要的工作空间大于通过参数提供的工作空间，将通过流分配器分配和释放工作空间。
    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>;
}

macro_rules! op_trait {
    ($name:ident $($body:item)*) => {
        pub trait $name<H: $crate::Hardware>:
            $crate::Operator<
            Hardware = H,
            TopoNode = H,
            Args = Args<H>,
        >{$($body)*}
    };
}

macro_rules! comm_trait {
    ($name:ident $($body:item)*) => {
        pub trait $name<H: $crate::Hardware, N: $crate::TopoNode<H>>:
            $crate::Operator<
            Hardware = H,
            TopoNode = N,
            Args = Args<H>,
        >{$($body)*}
    };
}

macro_rules! non_comm {
    ($name:ident impl $trait:ident) => {
        pub type $name<H, R> = crate::NonComm<H, R, Args<H>>;
        impl<H, R> $trait<H, H> for $name<H, R>
        where
            H: crate::Hardware,
            R: crate::rearrange::Rearrange<H>,
        {
        }
    };
}

pub(crate) use {comm_trait, non_comm, op_trait};

#[repr(transparent)]
pub struct NonComm<H, R, A>(R, PhantomData<(H, A)>);

impl<H, R, A> Operator for NonComm<H, R, A>
where
    H: Hardware,
    R: Rearrange<H>,
    A: AsRef<rearrange::Args<H>>,
{
    type Hardware = H;
    type TopoNode = H;
    type Args = A;

    #[inline]
    fn new(node: &Self::TopoNode) -> Self {
        Self(R::new(node), PhantomData)
    }

    #[inline]
    fn scheme(
        &mut self,
        args: &Self::Args,
        max_workspace_size: usize,
    ) -> Result<usize, crate::SchemeError> {
        self.0.scheme(args.as_ref(), max_workspace_size)
    }

    #[inline]
    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), crate::LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let args = args.as_ref();
        if !addr_eq(args.dst_base, args.src_base) {
            self.0.launch(args, workspace, queue_alloc)?
        }
        Ok(())
    }
}
