use super::contexts;
use cublas::{Cublas, CublasSpore};
use cuda::{AsRaw, ContextResource, ContextSpore, Stream};
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::null_mut,
    sync::{
        atomic::{
            AtomicPtr,
            Ordering::{Acquire, Release},
        },
        OnceLock,
    },
};

pub fn use_cublas(stream: &Stream, f: impl FnOnce(&Cublas)) {
    let ctx = stream.ctx();
    let idx = contexts()
        .iter()
        .enumerate()
        .find(|(_, context)| unsafe { context.as_raw() == ctx.as_raw() })
        .expect("Use primary context")
        .0;

    let pool = &pools()[idx];
    let cublas = pool
        .pop()
        .map_or_else(|| Cublas::new(ctx), |spore| spore.sprout(ctx));

    cublas.set_stream(stream);
    f(&cublas);

    pool.push(cublas.sporulate());
}

fn pools() -> &'static [Pool<CublasSpore>] {
    static POOL: OnceLock<Vec<Pool<CublasSpore>>> = OnceLock::new();
    POOL.get_or_init(|| (0..cuda::Device::count()).map(|_| Pool::new()).collect())
}

struct Pool<T: Unpin>(AtomicPtr<Item<T>>);

struct Item<T> {
    value: T,
    next: *mut Item<T>,
}

impl<T: Unpin> Pool<T> {
    #[inline]
    fn new() -> Self {
        Self(AtomicPtr::new(null_mut()))
    }

    #[inline]
    fn update(&self, current: *mut Item<T>, new: *mut Item<T>) -> Option<*mut Item<T>> {
        match self.0.compare_exchange_weak(current, new, Release, Acquire) {
            Ok(_) => None,
            Err(current) => Some(current),
        }
    }

    fn push(&self, value: T) {
        let item = unsafe { alloc(Layout::new::<Item<T>>()) } as *mut Item<T>;
        unsafe {
            item.write(Item {
                value,
                next: self.0.load(Acquire),
            })
        };
        while let Some(current) = self.update(unsafe { (*item).next }, item) {
            unsafe { (*item).next = current };
        }
    }

    fn pop(&self) -> Option<T> {
        let mut item = self.0.load(Acquire);
        while !item.is_null() {
            if let Some(current) = self.update(item, unsafe { (*item).next }) {
                item = current;
            } else {
                break;
            }
        }

        if item.is_null() {
            None
        } else {
            let Item { value, .. } = unsafe { item.read() };
            unsafe { dealloc(item as _, Layout::new::<Item<T>>()) };
            Some(value)
        }
    }
}

impl<T: Unpin> Drop for Pool<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}
