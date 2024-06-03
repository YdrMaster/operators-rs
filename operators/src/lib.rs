mod data;
mod devices;
mod error;
mod tensor;

pub mod operators;

#[macro_use]
extern crate log;

pub use data::{types::*, DataLayout};
pub use error::ErrorPosition;
pub use tensor::TensorLayout;

pub trait Device {
    type Byte;
    type Context;
}

pub trait Operator<D: Device>: Sized {
    type Config;
    type ConfigError;

    fn config(config: Self::Config) -> Result<Self, Self::ConfigError>;

    type Kernel: Kernel<D>;
    type LoadError;

    fn load(&self, ctx: &D::Context) -> Result<Self::Kernel, Self::LoadError>;
}

pub trait Kernel<D> {
    type Scheme;
    type Config;
    type SchemeError;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError>;
}
