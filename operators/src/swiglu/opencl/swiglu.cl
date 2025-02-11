#define CL_TARGET_OPENCL_VERSION 300
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

typedef unsigned int Tidx;

__kernel void swiglu(
    __global Tval *gate,
    int const stride_gate,
    __global Tval *up,
    int const strid_up) {

    Tidx g_idx = get_global_id(0);
    Tidx g_idy = get_global_id(1);

    Tidx i = g_idx * stride_gate + g_idy;
    Tidx j = g_idx * strid_up + g_idy;

    Tval x = gate[i];
    Tval y = up[j];

    Tval sig = 1.0f / (1.0f + exp(-x));
    gate[i] = x * sig * y;
}
