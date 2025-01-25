template<class Tdata>
static __device__ void _add(
    Tdata *__restrict__ c,
    Tdata const *__restrict__ a,
    Tdata const *__restrict__ b) {
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
