#define CL_TARGET_OPENCL_VERSION 300
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rope_f32(
    __global float2 *t,
    unsigned int stride_token,
    unsigned int stride_head,
    __global const unsigned int *pos,
    const float theta) {

    int nh_l = get_local_size(0);
    int dh = get_local_size(1);
    int it = get_group_id(0);
    int ih_h = get_group_id(1);
    int ih_l = get_local_id(0);
    int ih = ih_h * nh_l + ih_l;
    int i = get_local_id(1);

    __global float2 *t2 = t + it * stride_token + ih * stride_head + i;

    float2 data = *t2;
    float angle = (float) (pos[it]) / pow(theta, (float) i / (float) dh);
    float sin_val = native_sin(angle);
    float cos_val = native_cos(angle);

    float2 result;
    result.x = data.x * cos_val - data.y * sin_val;
    result.y = data.x * sin_val + data.y * cos_val;
    *t2 = result;
}
