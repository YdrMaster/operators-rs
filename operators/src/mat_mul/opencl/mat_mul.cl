#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TILE_SIZE 32
#define K_SIZE 64
#define BLOCKSIZE 512

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

    __global float *p = C + global_id_x * cs + row_id * crs + cow_id * ccs;
    float valueC = *p;
    *p = beta * valueC + alpha * value;
}
