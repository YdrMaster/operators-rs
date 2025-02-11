#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rearrange(
    __global unsigned int *dst,
    unsigned int rsa,
    unsigned int csa,
    __global unsigned int *src,
    unsigned int rsb,
    unsigned int csb,
    unsigned int ncols,
    unsigned int unit) {

    int g_id = get_global_id(0);
    int group_id = g_id / unit;
    int l_id = g_id % unit;

    int rows = group_id / ncols;
    int cols = group_id % ncols;

    int i = (rows * rsa + cols * csa) * unit + l_id;
    int j = (rows * rsb + cols * csb) * unit + l_id;
    dst[i] = src[j];
}
