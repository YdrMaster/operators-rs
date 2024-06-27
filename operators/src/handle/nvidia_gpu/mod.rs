mod module;

use common::Pool;
use cublas::{CublasLtSpore, CublasSpore};
use cuda::{bindings::nvrtcResult, Context, ContextSpore, Device, Version};
use module::cache_ptx;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

pub(crate) use module::ModuleBox;

pub struct Handle(pub(crate) Arc<Internal>);

impl common::Handle for Handle {
    type Byte = cuda::DevByte;
    type Queue<'ctx> = cuda::Stream<'ctx>;
}

impl Handle {
    pub fn new(context: Context) -> Self {
        Self(Arc::new(Internal {
            context,
            cublas: Default::default(),
            cublas_lt: Default::default(),
            modules: Default::default(),
        }))
    }
}

pub(crate) struct Internal {
    context: Context,
    cublas: Pool<CublasSpore>,
    cublas_lt: Pool<CublasLtSpore>,
    modules: RwLock<HashMap<Key, Weak<ModuleBox>>>,
}

type Key = (String, Version);

impl Internal {
    #[inline]
    pub fn device(&self) -> Device {
        self.context.device()
    }

    pub fn compile(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        cc: Version,
        code: impl FnOnce() -> String,
    ) -> Result<Arc<ModuleBox>, (nvrtcResult, String)> {
        let key = (name.as_ref().to_string(), cc);
        let module = self
            .modules
            .read()
            .unwrap()
            .get(&key)
            .and_then(|m| m.upgrade());
        match module {
            Some(module) => Ok(module),
            None => {
                let ptx = cache_ptx(&key, code)?;
                let module = ModuleBox::share(self.clone(), key.clone(), &ptx);
                let _ = self
                    .modules
                    .write()
                    .unwrap()
                    .insert(key, Arc::downgrade(&module));
                Ok(module)
            }
        }
    }
}

impl Drop for Internal {
    fn drop(&mut self) {
        assert!(self.modules.read().unwrap().is_empty());
        self.context.apply(|ctx| {
            while let Some(cublas) = self.cublas.pop() {
                drop(cublas.sprout(ctx));
            }
            while let Some(cublas) = self.cublas_lt.pop() {
                drop(cublas.sprout(ctx));
            }
        });
    }
}
