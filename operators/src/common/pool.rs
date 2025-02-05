use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::null_mut,
    sync::atomic::{
        AtomicPtr,
        Ordering::{Acquire, Release},
    },
};

#[repr(transparent)]
pub struct Pool<T: Unpin>(AtomicPtr<Item<T>>);

struct Item<T> {
    value: T,
    next: *mut Item<T>,
}

impl<T: Unpin> Default for Pool<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Unpin> Pool<T> {
    #[inline]
    pub fn new() -> Self {
        Self(AtomicPtr::new(null_mut()))
    }

    #[inline]
    fn update(&self, current: *mut Item<T>, new: *mut Item<T>) -> Option<*mut Item<T>> {
        self.0
            .compare_exchange_weak(current, new, Release, Acquire)
            .err()
    }

    pub fn push(&self, value: T) {
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

    pub fn pop(&self) -> Option<T> {
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
