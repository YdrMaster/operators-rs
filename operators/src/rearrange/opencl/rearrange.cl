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

// __kernel void rearrange_fortest(
//     __global unsigned int *dst,
//     unsigned int rsa,
//     unsigned int csa,
//     __global unsigned int *src,
//     unsigned int rsb,
//     unsigned int csb,
//     unsigned int r,
//     unsigned int c,
//     unsigned int items) {

//     for (int m = 0; m < r; m++) {
//         for (int n = 0; n < c; n++) {
//             for (int p = 0; p < items; p++) {
//                 int i = (m * rsa + n * csa) * items + p;
//                 int j = (m * rsb + n * csb) * items + p;
//                 dst[i] = src[j];
//             }
//         }
//     }
// }