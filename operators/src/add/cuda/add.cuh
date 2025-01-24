template<class Tdata>
static __device__ void add(
    Tdata *__restrict__ c,
    Tdata const *__restrict__ a,
    Tdata const *__restrict__ b,
    int const stride) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * stride;
    c[idx] = a[idx] + b[idx];
}
