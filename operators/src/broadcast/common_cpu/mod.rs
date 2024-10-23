use super::{args::Meta, Args, Broadcast};
use crate::{
    common_cpu::{Cpu, InprocNode},
    rearrange, ByteOf, LaunchError, QueueAlloc, SchemeError, TopoNode,
};
use std::ptr::{addr_eq, copy, copy_nonoverlapping};

pub struct Operator(InprocNode<usize>);

impl Broadcast<Cpu, InprocNode<usize>> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = InprocNode<usize>;
    type Args = Args<Cpu>;

    fn new(node: &Self::TopoNode) -> Self {
        assert!(node.group_size().is_power_of_two());
        Self(node.clone())
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let rank = self.0.rank();
        let group_size = self.0.group_size();

        let Meta { size } = args.meta()?;
        let &Args {
            pair: rearrange::Args {
                dst_base, src_base, ..
            },
            root,
            ..
        } = args;

        let _guard = self.0.wait();
        if rank == root {
            for i in 0..group_size {
                if i != rank {
                    self.0.send(i, src_base as _)
                }
            }
            if !addr_eq(dst_base, src_base) {
                unsafe { copy(src_base, dst_base, size) }
            }
            for _ in 0..group_size - 1 {
                assert_eq!(self.0.recv(), usize::MAX)
            }
        } else {
            unsafe { copy_nonoverlapping(self.0.recv() as _, dst_base, size) }
            self.0.send(root, usize::MAX)
        }
        Ok(())
    }
}

#[test]
fn test_comm() {
    use crate::{common_cpu::ThisThread, Operator as _, TensorLayout};
    use digit_layout::types::U32;

    InprocNode::new(4)
        .into_iter()
        .map(|node| {
            std::thread::spawn(move || {
                let mut buf = [node.rank() as u32; 8];
                let op = Operator::new(&node);
                op.launch(
                    &Args {
                        pair: rearrange::Args {
                            dst_layout: TensorLayout::new_contiguous(U32, &[8]),
                            dst_base: buf.as_mut_ptr().cast(),
                            src_layout: TensorLayout::new_contiguous(U32, &[8]),
                            src_base: buf.as_ptr().cast(),
                        },
                        root: 1,
                    },
                    &mut [],
                    &ThisThread,
                )
                .unwrap();
                buf
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|h| assert_eq!(h.join().unwrap(), [1; 8]))
}
