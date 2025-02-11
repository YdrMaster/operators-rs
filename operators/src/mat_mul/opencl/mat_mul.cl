#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

#ifdef USE_HALF
#define MUL(valueA, valueB) (float) (valueA * valueB)
#define SCAL(beta, p, alpha, value) (half)(beta * (float) (*p) + alpha * value)
#else
#define MUL(valueA, valueB) valueA *valueB
#define SCAL(beta, p, alpha, value) beta *(*p) + alpha *value
#endif

__kernel void general_gemm(__global Tval *A, __global Tval *B, __global Tval *C,
                           int as, int ars, int acs, int bs, int brs, int bcs,
                           int cs, int crs, int ccs, int batch,
                           int M, int N, int K, float alpha, float beta) {
    int g_idx = get_global_id(0);
    int g_idy = get_global_id(1);
    int row_id = g_idy / N;
    int col_id = g_idy % N;

    Tval valueA = 0.0f;
    Tval valueB = 0.0f;
    float value = 0.0f;

    for (int i = 0; i < K; i++) {
        valueA = *(A + g_idx * as + row_id * ars + i * acs);
        valueB = *(B + g_idx * bs + i * brs + col_id * bcs);
        value += MUL(valueA, valueB);
    }

    __global Tval *p = C + g_idx * cs + row_id * crs + col_id * ccs;
    *p = SCAL(beta, p, alpha, value);
}
