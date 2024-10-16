use super::Cpu;
use crate::TopoNode;
use std::sync::{
    mpsc::{channel, Receiver, Sender},
    Arc, Mutex,
};

pub struct InprocNode<T> {
    rank: usize,
    senders: Arc<[Sender<T>]>,
    receiver: Arc<Mutex<Receiver<T>>>,
}

impl<T> Clone for InprocNode<T> {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            senders: self.senders.clone(),
            receiver: self.receiver.clone(),
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
        let server: Arc<[Sender<T>]> = senders.into();
        receivers
            .into_iter()
            .enumerate()
            .map(|(rank, receiver)| InprocNode {
                rank,
                senders: server.clone(),
                receiver,
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
