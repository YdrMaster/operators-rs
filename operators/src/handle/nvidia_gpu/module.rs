use super::{Internal, Key};
use cuda::{
    bindings::nvrtcResult, ComputeCapability, ContextResource, ContextSpore, Dim3, ModuleSpore,
    Ptx, Stream,
};
use log::warn;
use std::{
    collections::{hash_map::Entry::Occupied, HashMap},
    ffi::{c_void, CStr},
    mem::replace,
    ptr::addr_eq,
    sync::{Arc, OnceLock, RwLock},
};

pub(crate) fn cache_ptx(
    key: &Key,
    code: impl FnOnce() -> String,
) -> Result<Arc<Ptx>, (nvrtcResult, String)> {
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

pub(crate) struct ModuleBox {
    handle: Arc<Internal>,
    key: Key,
    module: Option<ModuleSpore>,
}

impl ModuleBox {
    pub(super) fn share(handle: Arc<Internal>, key: Key, ptx: &Ptx) -> Arc<Self> {
        let module = handle.context.apply(|ctx| ctx.load(ptx).sporulate());
        Arc::new(Self {
            handle,
            key,
            module: Some(module),
        })
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
        self.module
            .as_ref()
            .unwrap()
            .sprout_ref(stream.ctx())
            .get_kernel(name)
            .launch(grid_dims, block_dims, params, shared_mem, Some(stream))
    }
}

impl Drop for ModuleBox {
    #[inline]
    fn drop(&mut self) {
        let key = replace(
            &mut self.key,
            (String::new(), ComputeCapability { major: 0, minor: 0 }),
        );
        if let Occupied(entry) = self.handle.modules.write().unwrap().entry(key) {
            if addr_eq(entry.get().as_ptr(), self as *const _) {
                entry.remove();
            }
        }
        if let Some(module) = self.module.take() {
            self.handle.context.apply(|ctx| drop(module.sprout(ctx)));
        }
    }
}
