use crate::{Alloc, Hardware, Pool, QueueAlloc, QueueOf, SchemeCacheSize, SchemeDiversity};
use clrt::{BuildError, CommandQueue, Context, Kernel, Program, SvmBlob, SvmByte};
use lru::LruCache;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    fmt,
    hash::Hash,
    sync::Mutex,
};

pub struct ClDevice {
    ctx: Context,
    cache_size: SchemeCacheSize,
}

impl Hardware for ClDevice {
    type Byte = SvmByte;
    type Queue<'ctx> = CommandQueue;
}

impl ClDevice {
    #[inline]
    pub fn new(context: Context, cache_size: SchemeCacheSize) -> Self {
        Self {
            ctx: context,
            cache_size,
        }
    }

    #[inline]
    pub(crate) fn context(&self) -> &Context {
        &self.ctx
    }

    #[inline]
    pub fn new_cache<K: Hash + Eq, V>(&self, level: SchemeDiversity) -> Mutex<LruCache<K, V>> {
        self.cache_size.new_cache(level)
    }
}

impl Alloc<SvmBlob> for Context {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: SvmBlob) {}
}

impl Alloc<SvmBlob> for CommandQueue {
    #[inline]
    fn alloc(&self, size: usize) -> SvmBlob {
        self.ctx().malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, mem: SvmBlob) {
        self.free(mem, None)
    }
}

impl QueueAlloc for CommandQueue {
    type Hardware = ClDevice;
    type DevMem = SvmBlob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}

pub(crate) struct KernelCache {
    program: Program,
    kernels: HashMap<String, Pool<Kernel>>,
}

pub(crate) const CL2_0: &CStr = c"-cl-std=CL2.0";

pub struct CodeGen {
    code: &'static str,
    defines: Vec<(&'static str, String)>,
}

impl CodeGen {
    pub fn new(code: &'static str) -> Self {
        Self {
            code,
            defines: Default::default(),
        }
    }

    pub fn define(&mut self, name: &'static str, value: impl ToString) -> &mut Self {
        self.defines.push((name, value.to_string()));
        self
    }
}

impl fmt::Display for CodeGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, value) in &self.defines {
            writeln!(f, "#define {} {}", name, value)?
        }
        write!(f, "{}", self.code)
    }
}

impl KernelCache {
    pub fn new(ctx: &Context, src: &str, opts: &CStr) -> Self {
        let program = match ctx.build_from_source(src, opts) {
            Ok(program) => program,
            Err(BuildError::BuildFailed(log)) => {
                println!("{log}");
                panic!("Failed to build cl kernels")
            }
            Err(BuildError::Others(err)) => {
                panic!("Failed to build cl kernels with error {err}")
            }
        };
        let kernels = program
            .kernels()
            .into_iter()
            .map(|k| {
                let name = k.name();
                let pool = Pool::new();
                pool.push(k);
                (name, pool)
            })
            .collect();
        Self { program, kernels }
    }

    pub fn take(&self, name: &str) -> Option<Kernel> {
        self.kernels
            .get(name)?
            .pop()
            .or_else(|| self.program.get_kernel(CString::new(name).unwrap()))
    }

    pub fn put(&self, name: &str, kernel: Kernel) {
        self.kernels.get(name).unwrap().push(kernel)
    }
}
