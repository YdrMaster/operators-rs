#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

// assert: GROUP_SIZE is power of 2
#ifndef GROUP_SIZE
#define GROUP_SIZE 512
#endif

typedef unsigned int Tidx;

float group_sum(local float *data, float reg) {
    Tidx const idx = get_local_id(0),
               len = get_local_size(0);

    data[idx] = reg;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (Tidx stride = len >> 1; stride; stride >>= 1) {
        if (idx < stride) data[idx] += data[idx + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return data[0];
}


kernel void rms_norm_padding(
    global float *y,
    const int y_stride,
    global const float *x,
    const int x_stride,
    global const float *w,
    const float epsilon) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    int idx_x = group_id * x_stride + local_id;
    int idx_y = group_id * y_stride + local_id;

    float val_x = x[idx_x],
          val_w = w[local_id],
          squared = val_x * val_x;

    local float shared[GROUP_SIZE];
    float rms = native_rsqrt(group_sum(shared, squared) / local_size + epsilon);

    y[idx_y] = rms * val_x * val_w;
}

#define TILE_SIZE 16

kernel void rms_norm_folding(
    global float *y,
    const int y_stride,
    global const float *x,
    const int x_stride,
    global const float *w,
    const float epsilon,
    const int d) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    int items = (d + local_size - 1) / local_size;

    int idx_x = group_id * x_stride + local_id * items;
    int idx_y = group_id * y_stride + local_id * items;
    int idx_w = local_id * items;

    float
        val_x[TILE_SIZE],
        val_w[TILE_SIZE],
        squared = 0.0f;
    for (int i = 0; i < items; i++) {
        val_x[i] = (local_id * items + i < d) ? x[idx_x + i] : 0.0f;
        val_w[i] = (local_id * items + i < d) ? w[idx_w + i] : 0.0f;
        squared += val_x[i] * val_x[i];
    }

    local float shared[GROUP_SIZE];
    float rms = native_rsqrt(group_sum(shared, squared) / d + epsilon);

    for (int i = 0; i < items; i++) {
        if (local_id * items + i < d)
            y[idx_y + i] = rms * val_x[i] * val_w[i];
    }
}

kernel void rms_norm_general(
    global float *y,
    const int y_stride,
    global const float *x,
    const int x_stride,
    global const float *w,
    const float epsilon,
    const int d) {

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    int items = (d + local_size - 1) / local_size;

    int idx_x = group_id * x_stride + local_id * items;
    int idx_y = group_id * y_stride + local_id * items;
    int idx_w = local_id * items;

    float squared = 0.0f;
    for (int i = 0; i < items; i++) {
        float val_x = (local_id * items + i < d) ? x[idx_x + i] : 0.0f;
        squared += val_x * val_x;
    }

    local float shared[GROUP_SIZE];
    float rms = native_rsqrt(group_sum(shared, squared) / d + epsilon);

    for (int i = 0; i < items; i++) {
        float val_x = (local_id * items + i < d) ? x[idx_x + i] : 0.0f,
              val_w = (local_id * items + i < d) ? w[idx_w + i] : 0.0f;
        if ((local_id * items + i) < d)
            y[idx_y + i] = rms * val_x * val_w;
    }
}
