pub trait Unsigned {
    fn from(u: usize) -> Self;
    fn val(&self) -> usize;
}

macro_rules! impl_idx {
    ($( $ty:ty )+) => {
        $(
            impl Unsigned for $ty {
                #[inline]
                fn from(u: usize) -> Self {
                    u as _
                }

                #[inline]
                fn val(&self) -> usize {
                    *self as _
                }
            }
        )+
    };
}

impl_idx! { u8 u16 u32 u64 u128 usize }
