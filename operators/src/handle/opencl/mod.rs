use crate::{Alloc, Hardware, Pool, QueueAlloc, QueueOf};
use clrt::{BuildError, CommandQueue, Context, Kernel, Program, SvmBlob, SvmByte};
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
};

#[repr(transparent)]
pub struct ClDevice(Context);

impl Hardware for ClDevice {
    type Byte = SvmByte;
    type Queue<'ctx> = CommandQueue;
}

impl ClDevice {
    #[inline]
    pub fn new(context: Context) -> Self {
        Self(context)
    }

    #[inline]
    pub(crate) fn context(&self) -> &Context {
        &self.0
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

    pub fn get_kernel(&self, name: &str) -> Option<Kernel> {
        self.kernels
            .get(name)?
            .pop()
            .or_else(|| self.program.get_kernel(CString::new(name).unwrap()))
    }

    pub fn set_kernel(&self, name: &str, kernel: Kernel) {
        self.kernels.get(name).unwrap().push(kernel)
    }
}
