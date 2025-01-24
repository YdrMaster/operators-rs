template<class Tdata, class Tidx>
static __device__ void add_rows(
    Tdata *__restrict__ dst,
    Tdata const *__restrict__ src,
    Tidx const *__restrict__ i,
    int const stride_d_b,
    int const stride_d_m,
    int const stride_s,
    int const stride_i) {
    auto idx_n = blockIdx.x * blockDim.x + threadIdx.x;
    auto idst = blockIdx.z * stride_d_b + blockIdx.y * stride_d_m + idx_n;
    auto isrc = i[blockIdx.z * stride_i + blockIdx.y] * stride_s + idx_n;
    dst[idst] += src[isrc];
}
