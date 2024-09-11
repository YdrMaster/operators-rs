template<class Tmem>
static __device__ void rearrange(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const ncols) {

    auto row = blockIdx.y,
         col = blockIdx.x * blockDim.y + threadIdx.y;
    if (col >= ncols) return;

    auto thread = threadIdx.x,
         warp_size = blockDim.x;
    auto i = (row * rsa + col * csa) * warp_size + thread;
    auto j = (row * rsb + col * csb) * warp_size + thread;
    // printf("%d %d %d %d: row = %d, col = %d, nrows = %d, ncols = %d, rsa = %d, rsb = %d, csa = %d, csb = %d, warp_size = %d, thread = %d, i = %d, j = %d\n",
    //        blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, row, col, gridDim.y, ncols, rsa, rsb, csa, csb, warp_size, thread, i, j);

    reinterpret_cast<Tmem *>(dst)[i] = reinterpret_cast<Tmem const *>(src)[j];
}
