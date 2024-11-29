#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchemeErrorKind {
    TypeNotSupport,
    TypeMismatch,
    RankNotSupport,
    RankMismatch,
    ShapeNotSupport,
    ShapeMismatch,
    StridesNotSupport,
    ArgsNotSupport,
    DynamicNotSupport,
}

#[derive(Clone, Debug)]
pub struct SchemeError {
    pub kind: SchemeErrorKind,
    pub info: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchErrorKind {
    Scheme(SchemeErrorKind),
    ExecutionFailed,
}

#[derive(Clone, Debug)]
pub struct LaunchError {
    pub kind: LaunchErrorKind,
    pub info: String,
}

impl From<SchemeError> for LaunchError {
    fn from(SchemeError { kind, info }: SchemeError) -> Self {
        Self {
            kind: LaunchErrorKind::Scheme(kind),
            info,
        }
    }
}

pub(super) mod functions {
    use super::{LaunchError, LaunchErrorKind::*, SchemeError, SchemeErrorKind::*};

    macro_rules! builder {
        ($ty:ident: $name:ident $kind:expr) => {
            #[inline]
            pub fn $name(info: impl Into<String>) -> $ty {
                $ty {
                    kind: $kind,
                    info: info.into(),
                }
            }
        };
    }

    builder!(SchemeError: type_not_support    TypeNotSupport   );
    builder!(SchemeError: type_mismatch       TypeMismatch     );
    builder!(SchemeError: rank_mismatch       RankMismatch     );
    builder!(SchemeError: rank_not_support    RankNotSupport   );
    builder!(SchemeError: shape_not_support   ShapeNotSupport  );
    builder!(SchemeError: shape_mismatch      ShapeMismatch    );
    builder!(SchemeError: strides_not_support StridesNotSupport);
    builder!(SchemeError: args_not_support    ArgsNotSupport   );
    builder!(SchemeError: dyn_not_support     DynamicNotSupport);

    builder!(LaunchError: execution_failed    ExecutionFailed  );
}
