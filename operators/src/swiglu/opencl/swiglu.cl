#define CL_TARGET_OPENCL_VERSION 300
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void swiglu(
    __global float *gate,
    const int stride_gate,
    __global float *up,
    const int strid_up) {

    //计算线程和块索引
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    //计算索引
    int i = global_id_x * stride_gate + global_id_y;
    int j = global_id_x * strid_up + global_id_y;

    //取值
    float x = gate[i];
    float y = up[j];

    //计算
    float sig = 1.0f / (1.0f + exp(-x));
    gate[i] = x * sig * y;
}