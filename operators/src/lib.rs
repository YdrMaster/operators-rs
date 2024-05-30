mod data;
mod devices;
mod error;
pub mod operators;
mod tensor;

pub use data::{types::*, DataLayout};
pub use error::ErrorPosition;
pub use tensor::TensorLayout;

pub trait Device {
    type Byte;
    type Context;
}

pub trait Operator: Sized {
    type Device: Device;

    type Config;
    type ConfigError;

    fn config(config: Self::Config) -> Result<Self, Self::ConfigError>;

    type Kernel: Kernel<Self::Device>;
    type LoadError;

    fn load(
        &self,
        ctx: &<Self::Device as Device>::Context,
    ) -> Result<Self::Kernel, Self::LoadError>;
}

pub trait Kernel<Device> {
    type Scheme;
    type Config;
    type SchemeError;

    fn scheme(&self, config: Self::Config) -> Result<Self::Scheme, Self::SchemeError>;
}
