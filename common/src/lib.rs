#![deny(warnings)]

mod argument;
mod error;
mod pool;
mod tensor;

pub use argument::{dyn_, ArgVal, Argument};
pub use error::ErrorPosition;
pub use pool::Pool;
pub use tensor::TensorLayout;

/// 算力硬件的上下文资源句柄。
pub trait Handle {
    /// 硬件直接访问的存储空间，通过字节占位符描述。
    type Byte;
    /// 硬件驱动提供的任务队列，带有生命周期约束。
    type Queue<'ctx>;
}

pub type ByteOf<D> = <D as Handle>::Byte;
pub type QueueOf<'ctx, D> = <D as Handle>::Queue<'ctx>;

pub trait Operator {
    type Handle: Handle;
    type Args;
    type SchemeError;
    type LaunchError;

    fn new(handle: &Self::Handle) -> Self;
    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError>;
    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError>;
}
