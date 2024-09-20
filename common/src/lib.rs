#![deny(warnings)]

mod argument;
mod error;
mod pool;
mod tensor;

use std::ops::DerefMut;

pub use argument::{dyn_, ArgVal, Argument};
use dev_mempool::Alloc;
pub use error::{
    functions::*, LaunchError, LaunchErrorKind, ParamError, ParamErrorKind, SchemeError,
    SchemeErrorKind,
};
pub use pool::Pool;
pub use tensor::TensorLayout;

/// 算力硬件的上下文资源句柄。
pub trait Handle {
    /// 硬件直接访问的存储空间，通过字节占位符描述。
    type Byte;
    /// 分配器产生的持有所有权的设备内存，被上下文的生命周期约束。
    type DevMem<'ctx>: DerefMut<Target = [Self::Byte]> + 'ctx;
    /// 硬件驱动提供的任务队列，带有生命周期约束。
    type Queue<'ctx>: Alloc<Self::DevMem<'ctx>>;
}

pub type ByteOf<D> = <D as Handle>::Byte;
pub type QueueOf<'ctx, D> = <D as Handle>::Queue<'ctx>;

pub trait Operator {
    type Handle: Handle;
    type Args;

    fn new(handle: &Self::Handle) -> Self;
    fn scheme(&mut self, args: &Self::Args) -> Result<(), SchemeError>;
    fn launch(&self, args: &Self::Args, queue: &QueueOf<Self::Handle>) -> Result<(), LaunchError>;
}
