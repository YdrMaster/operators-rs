mod library;
mod module;

use common::Pool;
use cublas::{Cublas, CublasLtSpore, CublasSpore};
use cuda::{Context, ContextResource, ContextSpore, CurrentCtx, Device, Stream, Version};
use libloading::Library;
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
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(Arc::new(Internal {
            context,
            cublas: Default::default(),
            cublas_lt: Default::default(),
            modules: Default::default(),
        }))
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        self.0.context.apply(f)
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

    pub fn cublas(&self, stream: &Stream, f: impl FnOnce(&Cublas)) {
        let cublas = self.cublas.pop().map_or_else(
            || Cublas::new(stream.ctx()),
            |cublas| cublas.sprout(stream.ctx()),
        );
        f(&cublas);
        self.cublas.push(cublas.sporulate());
    }

    pub fn compile_kernel(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        cc: Version,
        code: impl FnOnce() -> String,
    ) -> Arc<ModuleBox> {
        let key = (name.as_ref().to_string(), cc);
        let module = self
            .modules
            .read()
            .unwrap()
            .get(&key)
            .and_then(|m| m.upgrade());
        match module {
            Some(module) => module,
            None => {
                let module = ModuleBox::share(self.clone(), key.clone(), code);
                let _ = self
                    .modules
                    .write()
                    .unwrap()
                    .insert(key, Arc::downgrade(&module));
                module
            }
        }
    }

    #[allow(unused)]
    pub fn compile(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        cc: Version,
        code: impl FnOnce() -> String,
    ) -> Arc<Library> {
        library::cache_lib(&(name.as_ref().to_string(), cc), code).unwrap()
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
