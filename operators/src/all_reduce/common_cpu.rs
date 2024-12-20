use super::{args::Meta, AllReduce, Args, ReduceOp};
use crate::{
    broadcast::{self, common_cpu::Operator as Broadcast},
    common_cpu::{Cpu, InprocNode},
    rearrange, ByteOf, LaunchError, QueueAlloc, SchemeError, TopoNode,
};
use digit_layout::DigitLayout;
use half::{bf16, f16};
use std::{
    iter::zip,
    ops::AddAssign,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Operator {
    node: InprocNode<usize>,
    broadcast: Broadcast,
}

impl AllReduce<Cpu, InprocNode<usize>> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = InprocNode<usize>;
    type Args = Args<Cpu>;

    fn new(node: &Self::TopoNode) -> Self {
        assert!(node.group_size().is_power_of_two());
        Self {
            node: node.clone(),
            broadcast: Broadcast::new(node),
        }
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
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let rank = self.node.rank();
        let group_size = self.node.group_size();
        if group_size == 1 {
            return Ok(());
        }

        let Meta { dt, size } = args.meta()?;
        let &Args {
            pair: rearrange::Args {
                dst_base, src_base, ..
            },
            op,
        } = args;

        let mut ptr = src_base;
        let mut i = rank;
        let mut root = rank;
        let mut stride = 1;
        let guard = self.node.wait();
        while stride < group_size {
            if i % 2 == 0 {
                root += stride;
                self.node.send(root, ptr as _);

                i /= 2;
                stride *= 2;
                while stride < group_size {
                    if i % 2 == 0 {
                        root += stride;
                    }
                    i /= 2;
                    stride *= 2;
                }

                break;
            } else {
                reduce(dt, op, size, dst_base, self.node.recv() as _);
                ptr = dst_base;

                i /= 2;
                stride *= 2;
            }
        }
        drop(guard);
        self.broadcast.launch(
            &broadcast::Args {
                pair: args.pair.clone(),
                root,
            },
            workspace,
            queue_alloc,
        )
    }
}

fn reduce(dt: DigitLayout, op: ReduceOp, len: usize, buf: *mut u8, src: *const u8) {
    match op {
        ReduceOp::Sum => {
            macro_rules! sum {
                ($( $dt:ident => $ty:ty )+ ) => {
                    match dt {
                        $( digit_layout::types::$dt => sum::<$ty>(len, buf, src), )+
                        _ => todo!(),
                    }
                };
            }
            sum! {
                U8   => u8
                I8   => i8
                U16  => u16
                I16  => i16
                F16  => f16
                BF16 => bf16
                U32  => u32
                I32  => i32
                F32  => f32
                U64  => u64
                I64  => i64
                F64  => f64
                U128 => u128
                I128 => i128
            }
        }
        ReduceOp::Prod | ReduceOp::Min | ReduceOp::Max | ReduceOp::Mean => todo!(),
    }
}

fn sum<T: AddAssign + Clone>(len: usize, buf: *mut u8, src: *const u8) {
    let dst = unsafe { from_raw_parts_mut(buf.cast::<T>(), len) };
    let src = unsafe { from_raw_parts(src.cast::<T>(), len) };
    for (dst, src) in zip(dst, src) {
        *dst += src.clone();
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
                        op: ReduceOp::Sum,
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
        .for_each(|h| assert_eq!(h.join().unwrap(), [0 + 1 + 2 + 3; 8]));
}
