use lru::LruCache;
use std::{hash::Hash, num::NonZeroUsize, sync::Mutex};

#[derive(Clone, Debug)]
pub struct SchemeCacheSize {
    pub low: usize,
    pub medium: usize,
    pub high: usize,
}

impl Default for SchemeCacheSize {
    fn default() -> Self {
        Self {
            low: 4,
            medium: 16,
            high: 64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SchemeDiversity {
    Low,
    Medium,
    High,
}

impl SchemeCacheSize {
    pub fn new_cache<K: Hash + Eq, V>(&self, level: SchemeDiversity) -> Mutex<LruCache<K, V>> {
        let size = match level {
            SchemeDiversity::Low => self.low,
            SchemeDiversity::Medium => self.medium,
            SchemeDiversity::High => self.high,
        };
        Mutex::new(LruCache::new(NonZeroUsize::new(size).unwrap()))
    }
}
