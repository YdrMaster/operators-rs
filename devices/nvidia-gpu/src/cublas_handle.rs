use super::contexts;
use common::Pool;
use cublas::{Cublas, CublasSpore};
use cuda::{AsRaw, ContextResource, ContextSpore, Stream};
use std::sync::OnceLock;

#[inline]
pub fn pools() -> &'static [Pool<CublasSpore>] {
    static POOL: OnceLock<Vec<Pool<CublasSpore>>> = OnceLock::new();
    POOL.get_or_init(|| {
        contexts()
            .iter()
            .map(|context| {
                let pool = Pool::new();
                pool.push(context.apply(|ctx| Cublas::new(ctx).sporulate()));
                pool
            })
            .collect()
    })
}

pub fn use_cublas(stream: &Stream, f: impl FnOnce(&Cublas)) {
    let ctx = stream.ctx();
    let contexts = contexts();
    let idx = contexts
        .iter()
        .enumerate()
        .find(|(_, context)| unsafe { context.as_raw() == ctx.as_raw() })
        .expect("Use primary context")
        .0;

    let pool = &pools()[idx];
    let mut cublas = pool
        .pop()
        .map_or_else(|| Cublas::new(ctx), |spore| spore.sprout(ctx));

    cublas.set_stream(stream);
    f(&cublas);

    pool.push(cublas.sporulate());
}
