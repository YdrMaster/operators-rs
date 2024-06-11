#![deny(warnings)]

mod data;
mod error;
mod tensor;

pub use data::{types::*, DataLayout};
pub use error::ErrorPosition;
pub use tensor::TensorLayout;

/// 算力硬件的抽象。
pub trait Device {
    /// 硬件直接访问的存储空间，通过字节占位符描述。
    type Byte;

    /// 硬件驱动提供的任务队列，带有生命周期约束。
    type Queue<'ctx>;
}

pub type QueueOf<'ctx, D> = <D as Device>::Queue<'ctx>;

/// 算子的抽象。
///
/// 算子必须指定将由哪种算力硬件执行，即模板参数 `D`。
pub trait Operator<D: Device>: Sized {
    /// 配置算子时必要的静态参数。
    type Config;

    /// 配置算子过程中可能产生的错误。
    type Error;

    /// 创建算子实例。
    fn new(config: &Self::Config) -> Result<Self, Self::Error>;
}

/// 核函数执行方案的抽象。
///
/// 核函数执行方案是算子 `Op` 在硬件 `D` 上执行的具体方案，可能带有优化、生成或搜索机制。
pub trait Scheme<D: Device, Op: Operator<D>>: Sized {
    /// 算子执行的布局和其他属性。
    type LayoutAttrs;

    /// 创建执行方案过程中可能产生的错误。
    type Error;

    /// 创建执行方案实例。
    fn new(op: &Op, layout: Self::LayoutAttrs) -> Result<Self, Self::Error>;

    /// 发射核函数时需要的指针和其他参数，指针应指向 `<D as Device>::Byte`。
    type Params<'ctx>;

    /// 用当前方案将核函数发射到队列 `queue`。
    fn launch(&self, params: &Self::Params<'_>, queue: &QueueOf<D>);
}
