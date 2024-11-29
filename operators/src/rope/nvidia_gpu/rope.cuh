#include <cuda_fp16.h>

template<class Tp>
static __device__ void padding(
    half2 *__restrict__ t,
    int const stride_token,
    int const stride_head,
    Tp const *__restrict__ pos,
    float const theta) {

    auto const
        // nt = gridDim.y,
        // nh_h = gridDim.x,
        nh_l = blockDim.y,
        dh = blockDim.x,

        it = blockIdx.y,        // token index
        ih_h = blockIdx.x,      // head index (high)
        ih_l = threadIdx.y,     // head index (low)
        ih = ih_h * nh_l + ih_l,// head index
        i = threadIdx.x;        // element index

    t += it * stride_token + ih * stride_head + i;
    float a = t->x, b = t->y, sin, cos;
    sincosf(float(pos[it]) / powf(theta, float(i) / float(dh)), &sin, &cos);
    *t = half2(a * cos - b * sin, a * sin + b * cos);
}
