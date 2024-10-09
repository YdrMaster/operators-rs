use crate::{ByteOf, QueueAlloc};
use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

pub enum Workspace<'a, QA: QueueAlloc> {
    Ext(&'a mut [ByteOf<QA::Hardware>]),
    Int(ManuallyDrop<QA::DevMem>, &'a QA),
}

impl<'a, QA: QueueAlloc> Workspace<'a, QA> {
    #[inline]
    pub fn new(queue_alloc: &'a QA, ext: &'a mut [ByteOf<QA::Hardware>], size: usize) -> Self {
        if ext.len() >= size {
            Self::Ext(ext)
        } else {
            let dev_mem = queue_alloc.alloc(size);
            Self::Int(ManuallyDrop::new(dev_mem), queue_alloc)
        }
    }
}

impl<QA: QueueAlloc> Deref for Workspace<'_, QA> {
    type Target = [ByteOf<QA::Hardware>];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ext(ext) => ext,
            Self::Int(dev_mem, _) => dev_mem,
        }
    }
}

impl<QA: QueueAlloc> DerefMut for Workspace<'_, QA> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ext(ext) => ext,
            Self::Int(dev_mem, _) => dev_mem,
        }
    }
}

impl<QA: QueueAlloc> Drop for Workspace<'_, QA> {
    fn drop(&mut self) {
        match self {
            Self::Ext(_) => {}
            Self::Int(dev_mem, qa) => qa.free(unsafe { ManuallyDrop::take(dev_mem) }),
        }
    }
}

pub(crate) struct WorkspaceCollector {
    base: Vec<usize>,
    sub: usize,
}

impl WorkspaceCollector {
    #[inline]
    pub fn new() -> Self {
        Self {
            base: Vec::with_capacity(2),
            sub: 0,
        }
    }

    #[inline]
    pub fn push_base(&mut self, base: usize) {
        self.base.push(base)
    }

    #[inline]
    pub fn push_sub(&mut self, sub: usize) {
        self.sub = self.sub.max(sub)
    }

    pub fn cauculate(mut self, max_workspace_size: usize) -> usize {
        self.base.push(self.sub);
        let mut ans = 0;
        for s in self.base {
            if ans + s <= max_workspace_size {
                ans += s;
            } else {
                return ans;
            }
        }
        ans
    }
}
