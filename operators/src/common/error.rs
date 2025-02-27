#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchErrorKind {
    TypeNotSupport,
    TypeMismatch,
    RankNotSupport,
    RankMismatch,
    ShapeNotSupport,
    ShapeMismatch,
    StridesNotSupport,
    ArgsNotSupport,
    DynamicNotSupport,
    ExecutionFailed,
}

#[derive(Clone, Debug)]
pub struct LaunchError {
    pub kind: LaunchErrorKind,
    pub info: String,
}

pub(super) mod functions {
    use super::{LaunchError, LaunchErrorKind::*};

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

    builder!(LaunchError: type_not_support    TypeNotSupport   );
    builder!(LaunchError: type_mismatch       TypeMismatch     );
    builder!(LaunchError: rank_mismatch       RankMismatch     );
    builder!(LaunchError: rank_not_support    RankNotSupport   );
    builder!(LaunchError: shape_not_support   ShapeNotSupport  );
    builder!(LaunchError: shape_mismatch      ShapeMismatch    );
    builder!(LaunchError: strides_not_support StridesNotSupport);
    builder!(LaunchError: args_not_support    ArgsNotSupport   );
    builder!(LaunchError: dyn_not_support     DynamicNotSupport);

    builder!(LaunchError: execution_failed    ExecutionFailed  );
}
