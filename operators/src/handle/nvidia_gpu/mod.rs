mod library;
mod module;

use common::Pool;
use cublas::{Cublas, CublasLtSpore, CublasSpore};
use cuda::{Context, ContextResource, ContextSpore, CurrentCtx, Device, Stream, Version};
use digit_layout::DigitLayout;
use libloading::Library;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

pub(crate) use library::{EXPORT, EXPORT_H};
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

    #[cfg(test)]
    pub(crate) fn init() -> Option<Self> {
        if let Err(cuda::NoDevice) = cuda::init() {
            return None;
        }
        Some(Self::new(cuda::Device::new(0).context()))
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

    pub fn compile(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        cc: Version,
        code: impl FnOnce() -> String,
    ) -> Arc<Library> {
        library::cache_lib(&(name.as_ref().to_string(), cc), code)
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

pub(crate) fn dt_name(dt: DigitLayout) -> &'static str {
    use digit_layout::types as ty;
    match dt {
        ty::U8 => "unsigned char",
        ty::U16 => "unsigned short",
        ty::U32 => "unsigned int",
        ty::U64 => "unsigned long long",

        ty::I8 => "char",
        ty::I16 => "short",
        ty::I32 => "int",
        ty::I64 => "long long",

        ty::F16 => "half",
        ty::F32 => "float",
        ty::F64 => "double",
        ty::BF16 => "nv_bfloat16",

        ty::Bool => "bool",

        _ => panic!("Unknown digit layout: {dt:?}"),
    }
}

/// 并行转换类型并异步拷贝到显存。
#[cfg(test)]
pub(crate) fn cast_load<'ctx, T, U, F>(val: &[T], f: F, stream: &Stream<'ctx>) -> cuda::DevMem<'ctx>
where
    T: Sync,
    U: Send + Copy,
    F: Sync + Fn(&T) -> U,
{
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    let mut host = stream.ctx().malloc_host::<U>(val.len());
    let host = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), val.len()) };
    host.into_par_iter().zip(val).for_each(|(y, x)| *y = f(x));
    stream.from_host(host)
}
