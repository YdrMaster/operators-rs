#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TILE_SIZE 32
#define K_SIZE 64
#define BLOCKSIZE 512

__kernel void gemv_f32v2(__global float *A, __global float *B, __global float *C,
                         int as, int ars, int acs, int bs, int brs, int bcs,
                         int cs, int crs, int ccs, int batch,
                         int M, int N, int K, float alpha, float beta) {
    int localsize_x = get_local_size(0);
    int localsize_y = get_local_size(1);
    int groupid_x = get_group_id(0);
    int groupid_y = get_group_id(1);
    int localid_x = get_local_id(0);
    int localid_y = get_local_id(1);
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int global_size = get_global_size(1);
    int batch_id = global_id_y / (N * K);

    __local float localA[K_SIZE];
    float localB;

    localA[localid_y] = *(A + batch_id * as + global_id_x * ars + localid_y * acs);

    barrier(CLK_LOCAL_MEM_FENCE);

    localB = *(B + batch_id * bs + localid_y * brs);
    barrier(CLK_LOCAL_MEM_FENCE);
    localA[localid_y] = localA[localid_y] * localB;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = localsize_y / 2; offset > 0; offset /= 2) {
        if (localid_y < offset) {
            localA[localid_y] += localA[localid_y + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float value = localA[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    __global float *p = C + batch_id * cs + global_id_x * crs;
    float valueC = *p;
    *p = beta * valueC + alpha * value;
}

__kernel void gemv_f32(__global float *A, __global float *B, __global float *C,
                       int as, int ars, int acs, int bs, int brs, int bcs,
                       int cs, int crs, int ccs, int batch,
                       int M, int N, int K, float alpha, float beta) {
    int localsize_x = get_local_size(0);
    int localsize_y = get_local_size(1);
    int groupid_x = get_group_id(0);
    int groupid_y = get_group_id(1);
    int localid_x = get_local_id(0);
    int localid_y = get_local_id(1);
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int global_size = get_global_size(1);
    int batch_id = global_id_y / N;

    __local float localA[TILE_SIZE][16];
    __local float localB[16];

    float value = 0.0f;

    for (int i = 0; i < (K + 15) / 16; i++) {

        for (int j = 0; j < 16; j++) {
            localA[localid_x][j] = *(A + batch_id * as + global_id_x * ars +
                                     (i * 16 + j) * acs);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (localid_x < 16) {
            localB[localid_x] = *(B + batch_id * bs + (i * 16 + localid_x) * brs);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < 16; j++) {
            value += localA[localid_x][j] * localB[j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __global float *p = C + batch_id * cs + global_id_x * crs;
    float valueC = *p;
    *p = beta * valueC + alpha * value;
}

__kernel void general_gemm_f32(__global float *A, __global float *B, __global float *C,
                               int as, int ars, int acs, int bs, int brs, int bcs,
                               int cs, int crs, int ccs, int batch,
                               int M, int N, int K, float alpha, float beta) {
    int localid_x = get_local_id(0);
    int localid_y = get_local_id(1);
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int row_id = global_id_y / N;
    int cow_id = global_id_y % N;

    float valueA = 0.0f;
    float valueB = 0.0f;
    float value = 0.0f;
    for (int i = 0; i < K; i++) {

        valueA = *(A + global_id_x * as + row_id * ars + i * acs);
        valueB = *(B + global_id_x * bs + i * brs + cow_id * bcs);

        value += valueA * valueB;
    }

    float *p = C + global_id_x * cs + row_id * crs + cow_id * ccs;
    float valueC = *p;
    *p = beta * valueC + alpha * value;
}