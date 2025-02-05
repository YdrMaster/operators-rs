#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Ta
#define Ta float
#endif

#ifndef Tw
#define Tw float
#endif

// assert: GROUP_SIZE is power of 2
#ifndef GROUP_SIZE
#define GROUP_SIZE 512
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 1
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

kernel void rms_norm(
    global Ta *y_,
    int const y_stride,
    global Ta const *x_,
    int const x_stride,
    global Tw const *w,
    float const epsilon,
    Tidx const d) {

    Tidx g_idx = get_group_id(0),
         l_idx = get_local_id(0),
         l_len = get_local_size(0);
    global Ta
        *y = y_ + g_idx * y_stride;
    global Ta const
        *x = x_ + g_idx * x_stride;

    float
        val_x[ITEMS_THREAD],
        val_w[ITEMS_THREAD],
        squared = 0;
    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len) {
        val_x[i] = x[idx];
        val_w[i] = w[idx];
        squared += val_x[i] * val_x[i];
    }

    local float shared[GROUP_SIZE];
    float rms = native_rsqrt(group_sum(shared, squared) / d + epsilon);

    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len)
        y[idx] = rms * val_x[i] * val_w[i];
}
