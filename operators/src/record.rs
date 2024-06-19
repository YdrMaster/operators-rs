use std::sync::atomic::{
    AtomicBool,
    Ordering::{Relaxed, SeqCst},
};

static RECORD: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn start_record() {
    RECORD.store(true, Relaxed);
}

#[inline]
pub fn stop_record() {
    RECORD.store(false, SeqCst);
}

#[inline]
pub fn is_recording() -> bool {
    RECORD.load(Relaxed)
}
