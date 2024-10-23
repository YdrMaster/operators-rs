use super::Cpu;
use crate::TopoNode;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    mpsc::{channel, Receiver, Sender},
    Arc, Condvar, Mutex,
};

pub struct InprocNode<T> {
    rank: usize,
    senders: Box<[Sender<T>]>,
    receiver: Arc<Mutex<Receiver<T>>>,
    notifier: Arc<Notifier>,
    counter: Arc<AtomicUsize>,
}

impl<T> Clone for InprocNode<T> {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            senders: self.senders.clone(),
            receiver: self.receiver.clone(),
            notifier: self.notifier.clone(),
            counter: self.counter.clone(),
        }
    }
}

impl<T> InprocNode<T> {
    pub fn new(n: usize) -> Vec<InprocNode<T>> {
        let mut senders = Vec::with_capacity(n);
        let mut receivers = Vec::with_capacity(n);
        for _ in 0..n {
            let (sender, receiver) = channel();
            senders.push(sender);
            receivers.push(Arc::new(Mutex::new(receiver)));
        }
        let senders: Box<[Sender<T>]> = senders.into();
        let notifier = Arc::new(Notifier::new());

        receivers
            .into_iter()
            .enumerate()
            .map(|(rank, receiver)| InprocNode {
                rank,
                senders: senders.clone(),
                receiver,
                notifier: notifier.clone(),
                counter: Arc::new(AtomicUsize::new(0)),
            })
            .collect()
    }

    #[inline]
    pub(crate) fn send(&self, i: usize, msg: T) {
        self.senders[i].send(msg).unwrap();
    }

    #[inline]
    pub(crate) fn recv(&self) -> T {
        self.receiver.lock().unwrap().recv().unwrap()
    }

    #[must_use]
    #[inline]
    pub(crate) fn wait(&self) -> Guard {
        let count = self.counter.fetch_add(1, Relaxed) * self.group_size();
        self.notifier.wait(count)
    }
}

impl<T> TopoNode<Cpu> for InprocNode<T> {
    #[inline]
    fn processor(&self) -> &Cpu {
        &Cpu
    }
    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }
    #[inline]
    fn group_size(&self) -> usize {
        self.senders.len()
    }
}

#[repr(transparent)]
pub struct Guard<'a>(&'a Notifier);

impl Drop for Guard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.notify();
    }
}

struct Notifier {
    lock: Mutex<usize>,
    cond: Condvar,
}

impl Notifier {
    fn new() -> Self {
        Self {
            lock: Mutex::new(0),
            cond: Condvar::new(),
        }
    }

    fn wait(&self, count: usize) -> Guard {
        let _guard = self
            .cond
            .wait_while(self.lock.lock().unwrap(), |current| *current < count)
            .unwrap();
        Guard(self)
    }

    fn notify(&self) {
        *self.lock.lock().unwrap() += 1;
        self.cond.notify_all();
    }
}
