#![allow(unused)]

use common::Handle;

pub(crate) type MutPtr<H> = *mut <H as Handle>::Byte;
pub(crate) type ConstPtr<H> = *const <H as Handle>::Byte;

#[inline]
pub(crate) const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

macro_rules! get_or_err {
    ($name:ident) => {
        let Some(&$name) = $name.get_static() else {
            return Err(locate_error!());
        };
    };
}

macro_rules! op_trait {
    ($name:ident $($body:item)*) => {
        pub trait $name<H: common::Handle>:
            common::Operator<
            Handle = H,
            Args = Args<H>,
            SchemeError = common::ErrorPosition,
            LaunchError = common::ErrorPosition,
        >{$($body)*}
    };
}

pub(crate) use {get_or_err, op_trait};
