pub mod rms_norm;
pub mod rope;
pub mod swiglu;

#[allow(unused)]
#[inline]
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}
