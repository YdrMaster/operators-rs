use ascendcl::{Context, CurrentCtx, DevByte, Stream};
use std::sync::Arc;

pub struct Handle(pub(crate) Arc<Internal>);

impl common::Handle for Handle {
    type Byte = DevByte;
    type Queue<'ctx> = Stream<'ctx>;
}

impl Handle {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(Arc::new(Internal { context }))
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        self.0.context.apply(f)
    }
}

pub(crate) struct Internal {
    context: Context,
}
