use cndrv::{Context, ContextResource, ContextSpore, CurrentCtx, DevByte, Queue};
use cnnl::{Cnnl, CnnlSpore};
use common::Pool;
use std::sync::Arc;

pub struct Handle(pub(crate) Arc<Internal>);

impl common::Handle for Handle {
    type Byte = DevByte;
    type Queue<'ctx> = Queue<'ctx>;
}

impl Handle {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(Arc::new(Internal {
            context,
            cnnl: Default::default(),
        }))
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        self.0.context.apply(f)
    }
}

pub(crate) struct Internal {
    context: Context,
    cnnl: Pool<CnnlSpore>,
}

impl Internal {
    pub fn cnnl(&self, queue: &Queue, f: impl FnOnce(&Cnnl)) {
        let cnnl = self
            .cnnl
            .pop()
            .map_or_else(|| Cnnl::new(queue.ctx()), |cnnl| cnnl.sprout(queue.ctx()));
        f(&cnnl);
        self.cnnl.push(cnnl.sporulate());
    }
}

impl Drop for Internal {
    fn drop(&mut self) {
        self.context.apply(|ctx| {
            while let Some(cnnl) = self.cnnl.pop() {
                drop(cnnl.sprout(ctx));
            }
        });
    }
}
