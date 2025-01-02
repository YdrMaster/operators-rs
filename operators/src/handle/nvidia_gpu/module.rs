use super::{Handle, Key};
use cuda::{
    bindings::nvrtcResult, ContextResource, ContextSpore, CurrentCtx, Dim3, KernelFn, ModuleSpore,
    Ptx, Stream,
};
use log::warn;
use std::{
    collections::{hash_map::Entry::Occupied, HashMap},
    ffi::{c_void, CStr},
    ptr::addr_eq,
    sync::{Arc, OnceLock, RwLock},
};

pub(crate) struct ModuleBox {
    handle: Arc<Handle>,
    key: Key,
    module: Option<ModuleSpore>,
}

impl ModuleBox {
    pub(super) fn share(handle: Arc<Handle>, key: Key, code: impl FnOnce() -> String) -> Arc<Self> {
        let ptx = cache_ptx(&key, code).unwrap();
        let module = handle.context.apply(|ctx| ctx.load(&ptx).sporulate());
        Arc::new(Self {
            handle,
            key,
            module: Some(module),
        })
    }

    pub fn load<'ctx>(&'ctx self, name: impl AsRef<CStr>, ctx: &'ctx CurrentCtx) -> KernelFn<'ctx> {
        self.module
            .as_ref()
            .unwrap()
            .sprout_ref(ctx)
            .get_kernel(name)
    }

    pub fn launch(
        &self,
        name: impl AsRef<CStr>,
        grid_dims: impl Into<Dim3>,
        block_dims: impl Into<Dim3>,
        params: *const *const c_void,
        shared_mem: usize,
        stream: &Stream,
    ) {
        self.load(name, stream.ctx()).launch(
            grid_dims,
            block_dims,
            params,
            shared_mem,
            Some(stream),
        )
    }
}

impl Drop for ModuleBox {
    #[inline]
    fn drop(&mut self) {
        if let Occupied(entry) = self.handle.modules.write().unwrap().entry(self.key.clone()) {
            if addr_eq(entry.get().as_ptr(), self as *const _) {
                entry.remove();
            }
        }
        if let Some(module) = self.module.take() {
            self.handle.context.apply(|ctx| drop(module.sprout(ctx)));
        }
    }
}

fn cache_ptx(key: &Key, code: impl FnOnce() -> String) -> Result<Arc<Ptx>, (nvrtcResult, String)> {
    static CACHE: OnceLock<RwLock<HashMap<Key, Arc<Ptx>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);

    if let Some(ptx) = cache.read().unwrap().get(key) {
        return Ok(ptx.clone());
    }
    let (ptx, log) = Ptx::compile(code(), key.1);
    match ptx {
        Ok(ptx) => {
            if !log.is_empty() {
                warn!("{log}");
            }

            let ptx = Arc::new(ptx);
            let _ = cache.write().unwrap().insert(key.clone(), ptx.clone());
            Ok(ptx)
        }
        Err(e) => Err((e, log)),
    }
}
