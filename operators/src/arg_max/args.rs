use super::KVPair;
use crate::utils::{ConstPtr, MutPtr};
use common::{locate_error, ErrorPosition, Handle, TensorLayout, Workspace};
use digit_layout::DigitLayout;
use std::hash::{Hash, Hasher};

pub struct Args<H: Handle> {
    pub kv_pair: TensorLayout,
    pub kv_pair_base: MutPtr<H>,
    pub data: TensorLayout,
    pub data_base: ConstPtr<H>,
    pub workspace: Workspace<H>,
}

#[derive(PartialEq, Eq, Debug)]
pub(super) struct Meta {
    pub dt: DigitLayout,
    pub n: usize,
}

impl Hash for Meta {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dt.to_u32().hash(state);
        self.n.hash(state);
    }
}

impl<H: Handle> Args<H> {
    pub(super) fn meta(&self) -> Result<Meta, ErrorPosition> {
        if self.kv_pair.dt() != KVPair::<()>::LAYOUT {
            return Err(locate_error!("index must be KVpair"));
        }
        let &[n] = self.data.shape() else {
            return Err(locate_error!());
        };
        let Some(&n) = n.get_static() else {
            return Err(locate_error!("n must be static"));
        };
        Ok(Meta {
            dt: self.data.dt(),
            n,
        })
    }
}
