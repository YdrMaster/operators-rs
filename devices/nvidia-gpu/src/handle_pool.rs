use super::contexts;
use common::Pool;
use cublas::{Cublas, CublasSpore};
use cuda::{AsRaw, ContextResource, ContextSpore, Stream};
use std::sync::OnceLock;

pub fn use_cublas(stream: &Stream, f: impl FnOnce(&Cublas)) {
    let ctx = stream.ctx();
    let idx = contexts()
        .iter()
        .enumerate()
        .find(|(_, context)| unsafe { context.as_raw() == ctx.as_raw() })
        .expect("Use primary context")
        .0;

    static POOL: OnceLock<Vec<Pool<CublasSpore>>> = OnceLock::new();
    let pool = &POOL.get_or_init(|| (0..cuda::Device::count()).map(|_| Pool::new()).collect())[idx];
    let cublas = pool
        .pop()
        .map_or_else(|| Cublas::new(ctx), |spore| spore.sprout(ctx));

    cublas.set_stream(stream);
    f(&cublas);

    pool.push(cublas.sporulate());
}
