use std::alloc::Layout;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct DataLayout {
    packed: u8,
    signed: bool,
    exponent: u8,
    mantissa: u8,
}

impl DataLayout {
    #[inline]
    pub const fn new(packed: usize, signed: bool, exponent: usize, mantissa: usize) -> Self {
        assert!(packed <= u8::MAX as usize);
        assert!(exponent <= u8::MAX as usize);
        assert!(mantissa <= u8::MAX as usize);
        let total_bits = packed * (if signed { 1 } else { 0 } + exponent + mantissa);
        assert!(total_bits % u8::BITS as usize == 0);
        assert!(total_bits.is_power_of_two());
        Self {
            packed: packed as _,
            signed,
            exponent: exponent as _,
            mantissa: mantissa as _,
        }
    }

    #[inline]
    pub const fn packed(&self) -> usize {
        self.packed as _
    }

    #[inline]
    pub const fn signed(&self) -> bool {
        self.signed
    }

    #[inline]
    pub const fn exponent(&self) -> usize {
        self.exponent as _
    }

    #[inline]
    pub const fn mantissa(&self) -> usize {
        self.mantissa as _
    }

    #[inline]
    pub const fn bits(&self) -> usize {
        self.packed() * (if self.signed { 1 } else { 0 } + self.exponent() + self.mantissa())
    }

    #[inline]
    pub const fn layout(&self) -> Layout {
        let size = self.bits() / u8::BITS as usize;
        unsafe { Layout::from_size_align_unchecked(size, size) }
    }
}

pub(crate) mod types {

    #[macro_export]
    macro_rules! layout {
        ($name:ident i($bits:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, true, 0, $bits - 1);
        };
        ($name:ident u($bits:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            #[rustfmt::skip]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, false, 0, $bits);
        };
        ($name:ident e($exp:expr)m($mant:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, true, $exp, $mant);
        };

        ($name:ident i($bits:expr)) => {
            layout!($name i($bits)x(1));
        };
        ($name:ident u($bits:expr)) => {
            layout!($name u($bits)x(1));
        };
        ($name:ident e($exp:expr)m($mant:expr)) => {
            layout!($name e($exp)m($mant)x(1));
        };
    }

    layout!(I8     i( 8)          );
    layout!(I16    i(16)          );
    layout!(I32    i(32)          );
    layout!(I64    i(64)          );
    layout!(U8     u( 8)          );
    layout!(U16    u(16)          );
    layout!(U32    u(32)          );
    layout!(U64    u(64)          );
    layout!(F16    e(10)m( 5)     );
    layout!(BF16   e( 7)m( 8)     );
    layout!(F32    e(23)m( 8)     );
    layout!(F64    e(52)m(11)     );

    layout!(F16x2  e(10)m( 5)x(2) );
    layout!(BF16x2 e( 7)m( 8)x(2) );
}
