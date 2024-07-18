/// 可以和 f32 双向转换的类型；
pub trait BetweenF32 {
    /// 将 f32 转换为 Self；
    fn cast(f: f32) -> Self;
    /// 将 Self 转换为 f32；
    fn f32(&self) -> f32;
}

impl BetweenF32 for f32 {
    #[inline]
    fn cast(f: f32) -> Self {
        f
    }
    #[inline]
    fn f32(&self) -> f32 {
        *self
    }
}

impl BetweenF32 for half::f16 {
    #[inline]
    fn cast(f: f32) -> Self {
        Self::from_f32(f)
    }
    #[inline]
    fn f32(&self) -> f32 {
        Self::to_f32(*self)
    }
}
