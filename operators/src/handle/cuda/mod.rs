mod alloc;
mod library;
mod module;
#[cfg(use_nccl)]
mod nccl;

use crate::{Hardware, Pool, SchemeDiversity};
use cublas::{Cublas, CublasSpore};
use cuda::{
    self, AsRaw, Context, ContextResource, ContextSpore, CurrentCtx, Device, Stream, Version,
};
use digit_layout::DigitLayout;
use libloading::Library;
use lru::LruCache;
use std::{
    collections::HashMap,
    hash::Hash,
    num::NonZeroUsize,
    sync::{Arc, Mutex, RwLock, Weak},
};

pub(crate) use library::{EXPORT, EXPORT_H};
pub(crate) use module::ModuleBox;

#[cfg(use_nvidia)]
use cublas::CublasLtSpore;

#[cfg(use_nccl)]
pub use nccl::NcclNode;

pub use alloc::{MemPoolBlob, StreamMemPool};

pub struct Gpu(pub(crate) Arc<Handle>);

impl Hardware for Gpu {
    type Byte = cuda::DevByte;
    type Queue<'ctx> = cuda::Stream<'ctx>;
}

#[derive(Clone, Debug)]
pub struct Config {
    pub low_diversity_cache: usize,
    pub medium_diversity_cache: usize,
    pub high_diversity_cache: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            low_diversity_cache: 4,
            medium_diversity_cache: 16,
            high_diversity_cache: 64,
        }
    }
}

impl Gpu {
    #[inline]
    pub fn new(context: Context, config: Config) -> Self {
        Self(Arc::new(Handle {
            context,
            config,
            cublas: Default::default(),
            #[cfg(use_nvidia)]
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
        Some(Self::new(cuda::Device::new(0).context(), Config::default()))
    }
}

pub(crate) struct Handle {
    context: Context,
    #[allow(dead_code)]
    config: Config,
    cublas: Pool<CublasSpore>,
    #[cfg(use_nvidia)]
    cublas_lt: Pool<CublasLtSpore>,
    modules: RwLock<HashMap<Key, Weak<ModuleBox>>>,
}

type Key = (String, Version);

impl Handle {
    #[inline]
    pub fn device(&self) -> Device {
        self.context.device()
    }
    #[allow(dead_code)]
    pub fn cublas(&self, stream: &Stream, f: impl FnOnce(&Cublas)) {
        let ctx = stream.ctx();
        unsafe { assert_eq!(ctx.as_raw(), self.context.as_raw()) };

        let mut cublas = self
            .cublas
            .pop()
            .map_or_else(|| Cublas::new(ctx), |cublas| cublas.sprout(ctx));

        cublas.set_stream(stream);
        f(&cublas);

        self.cublas.push(cublas.sporulate());
    }
    #[allow(dead_code)]
    pub fn cublas_init(&self) {
        self.cublas.push(
            self.cublas
                .pop()
                .unwrap_or_else(|| self.context.apply(|ctx| Cublas::new(ctx).sporulate())),
        );
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
    #[allow(dead_code)]
    pub fn compile(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        cc: Version,
        code: impl FnOnce() -> String,
    ) -> Arc<Library> {
        library::cache_lib(&(name.as_ref().to_string(), cc), code)
    }
    #[allow(dead_code)]
    pub fn scheme_cache<K: Hash + Eq, V>(
        &self,
        diversity: SchemeDiversity,
    ) -> Mutex<LruCache<K, V>> {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(match diversity {
                SchemeDiversity::Low => self.config.low_diversity_cache,
                SchemeDiversity::Medium => self.config.medium_diversity_cache,
                SchemeDiversity::High => self.config.high_diversity_cache,
            })
            .unwrap(),
        ))
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        assert!(self.modules.read().unwrap().is_empty());
        self.context.apply(|ctx| {
            while let Some(cublas) = self.cublas.pop() {
                drop(cublas.sprout(ctx));
            }
            #[cfg(use_nvidia)]
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
    T: Sync + Copy,
    U: Send + Copy,
    F: Sync + Fn(T) -> U,
{
    let mut host = stream.ctx().malloc_host::<U>(val.len());
    let host = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), val.len()) };
    host.into_iter().zip(val).for_each(|(y, x)| *y = f(*x));

    #[cfg(use_nvidia)]
    let mem = stream.from_host(host);
    #[cfg(use_iluvatar)]
    let mem = stream.ctx().from_host(host);
    mem
}
