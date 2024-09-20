#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamErrorKind {
    TypeNotSupport,
    TypeMismatch,
    RankNotSupport,
    RankMismatch,
    ShapeNotSupport,
    ShapeMismatch,
    StridesNotSupport,
    DynamicNotSupport,
}

#[derive(Clone, Debug)]
pub struct ParamError {
    pub kind: ParamErrorKind,
    pub info: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchemeErrorKind {
    Param(ParamErrorKind),
}

#[derive(Clone, Debug)]
pub struct SchemeError {
    pub kind: SchemeErrorKind,
    pub info: String,
}

impl From<ParamError> for SchemeError {
    fn from(ParamError { kind, info }: ParamError) -> Self {
        Self {
            kind: SchemeErrorKind::Param(kind),
            info,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchErrorKind {
    Param(ParamErrorKind),
    SchemeNotSet,
    SchemeNotCompatible,
    OutOfWorkspace,
    ExecutionFailed,
}

#[derive(Clone, Debug)]
pub struct LaunchError {
    pub kind: LaunchErrorKind,
    pub info: String,
}

impl From<ParamError> for LaunchError {
    fn from(ParamError { kind, info }: ParamError) -> Self {
        Self {
            kind: LaunchErrorKind::Param(kind),
            info,
        }
    }
}

pub(super) mod functions {
    use super::{LaunchError, LaunchErrorKind::*, ParamError, ParamErrorKind::*};

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

    builder!(ParamError: type_not_support    TypeNotSupport   );
    builder!(ParamError: type_mismatch       TypeMismatch     );
    builder!(ParamError: rank_mismatch       RankMismatch     );
    builder!(ParamError: rank_not_support    RankNotSupport   );
    builder!(ParamError: shape_not_support   ShapeNotSupport  );
    builder!(ParamError: shape_mismatch      ShapeMismatch    );
    builder!(ParamError: strides_not_support StridesNotSupport);
    builder!(ParamError: dyn_not_support     DynamicNotSupport);

    builder!(LaunchError: scheme_not_set        SchemeNotSet       );
    builder!(LaunchError: scheme_not_compatible SchemeNotCompatible);
    builder!(LaunchError: out_of_workspace      OutOfWorkspace     );
    builder!(LaunchError: execution_failed      ExecutionFailed    );
}
