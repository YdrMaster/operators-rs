﻿use cuda::{
    AsRaw, Context, ContextResource, ContextSpore, Device as Gpu, Dim3, ModuleSpore, Ptx, Stream,
};
use std::{
    ffi::{c_void, CStr},
    sync::{OnceLock, RwLock},
    usize,
};

pub struct Device;

impl crate::Device for Device {
    type Byte = cuda::DevByte;
    type Context = std::sync::Arc<cuda::Context>;
}

#[allow(non_camel_case_types)]
pub(crate) struct __global__(usize);

impl __global__ {
    pub fn load(ptx: &Ptx) -> Self {
        let contexts = contexts();
        let modules = contexts
            .iter()
            .map(|context| context.apply(|ctx| ctx.load(ptx).sporulate()))
            .collect::<Vec<_>>();

        let mut library = module_library().write().unwrap();
        let index = library.len() / contexts.len();
        library.extend(modules);

        Self(index)
    }

    pub fn launch<T>(
        &self,
        name: impl AsRef<CStr>,
        grid_dims: impl Into<Dim3>,
        block_dims: impl Into<Dim3>,
        params: *const *const c_void,
        shared_mem: usize,
        stream: &Stream,
    ) {
        let contexts = contexts();
        let ctx = stream.ctx();
        let idx = contexts
            .iter()
            .enumerate()
            .find(|(_, context)| unsafe { context.as_raw() == ctx.as_raw() })
            .expect("Use primary context")
            .0;
        module_library().read().unwrap()[self.0 * contexts.len() + idx]
            .sprout_ref(ctx)
            .get_kernel(name.as_ref())
            .launch(grid_dims, block_dims, params, shared_mem, Some(stream));
    }
}

fn contexts() -> &'static [Context] {
    static CONTEXTS: OnceLock<Vec<Context>> = OnceLock::new();
    CONTEXTS.get_or_init(|| {
        cuda::init();
        (0..Gpu::count())
            .map(|i| Gpu::new(i as _).retain_primary())
            .collect()
    })
}

fn module_library() -> &'static RwLock<Vec<ModuleSpore>> {
    static MODULE_LIBRARY: OnceLock<RwLock<Vec<ModuleSpore>>> = OnceLock::new();
    MODULE_LIBRARY.get_or_init(Default::default)
}
