#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float2
#endif

#ifndef Tpos
#define Tpos unsigned int
#endif

#ifdef USE_HALF
#define LOAD_DATA(ptr) vload_half2(0, (__global half *) ptr)
#define STORE_DATA(ptr, val) vstore_half2(val, 0, (__global half *) ptr)
#else
#define LOAD_DATA(ptr) (*ptr)
#define STORE_DATA(ptr, val) (*ptr = val)
#endif

typedef unsigned int Tidx;

__kernel void rope(
    __global Tval *t,
    int const stride_token,
    int const stride_head,
    __global Tpos const *pos,
    float const theta) {

    Tidx nh_l = get_local_size(0),
         dh = get_local_size(1),
         it = get_group_id(0),
         ih_h = get_group_id(1),
         ih_l = get_local_id(0),
         ih = ih_h * nh_l + ih_l,
         i = get_local_id(1);

    __global Tval *t2 = t + it * stride_token + ih * stride_head + i;

    float2 data = LOAD_DATA(t2);
    float angle = (float) (pos[it]) / pow(theta, (float) i / (float) dh);
    float sin_val = native_sin(angle);
    float cos_val = native_cos(angle);

    float2 result;
    result.x = data.x * cos_val - data.y * sin_val;
    result.y = data.x * sin_val + data.y * cos_val;
    STORE_DATA(t2, result);
}
