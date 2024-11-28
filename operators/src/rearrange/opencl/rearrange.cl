#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rearrange(
    __global float *dst,
    unsigned int rsa,
    unsigned int csa,
    __global float *src,
    unsigned int rsb,
    unsigned int csb,
    unsigned int ncols,
    unsigned int items) {

    int local_size = get_local_size(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int block_size = local_size * items;

    int rows = group_id / ncols;
    int cols = group_id % ncols;

    for (int k = 0; k < items; k++) {
        int i = (rows * rsa + cols * csa) * block_size + local_id * items + k;
        int j = (rows * rsb + cols * csb) * block_size + local_id * items + k;
        dst[i] = src[j];
    }
}
