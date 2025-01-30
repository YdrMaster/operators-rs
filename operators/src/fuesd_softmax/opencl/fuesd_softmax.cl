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

float group_max(local float *data, float reg) {
    Tidx const idx = get_local_id(0),
               len = get_local_size(0);

    data[idx] = reg;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (Tidx stride = len >> 1; stride; stride >>= 1) {
        if (idx < stride) data[idx] = fmax(data[idx], data[idx + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return data[0];
}

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

kernel void softmax_padding(
    global Tval *att,
    Tidx const seq_len,
    Tidx const att_len) {

    Tidx local_id = get_local_id(0),
         group_id = get_group_id(0),
         global_id = group_id * att_len + local_id,
         local_size = get_local_size(0);

    float thread_data, max_val = -FLT_MAX, sum_val = 0;

    if (local_id < att_len)
        thread_data = (att_len + (group_id % seq_len) >= local_id + seq_len) ? att[global_id] : -FLT_MAX;
    else
        thread_data = -FLT_MAX;

    local float shared[GROUP_SIZE];

    max_val = group_max(shared, thread_data);

    thread_data = exp(thread_data - max_val);

    barrier(CLK_LOCAL_MEM_FENCE);
    sum_val = group_sum(shared, thread_data);

    if (local_id < att_len)
        att[global_id] = thread_data / sum_val;
}

#define ITEMS 16

kernel void softmax_folding(
    global Tval *att,
    Tidx const seq_len,
    Tidx const att_len) {

    Tidx local_id = get_local_id(0),
         group_id = get_group_id(0),
         global_id = get_global_id(0),
         local_size = get_local_size(0);

    Tidx items = (att_len + local_size - 1) / local_size,
         local_base = local_id * items,
         global_base = group_id * att_len + local_base;

    float thread_data[ITEMS], max_val = -FLT_MAX, sum_val = 0;

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data[i] = (att_len + group_id % seq_len >= (local_base + i) + seq_len)
                                 ? att[global_base + i]
                                 : -FLT_MAX;

            if (max_val < thread_data[i])
                max_val = thread_data[i];
        } else
            thread_data[i] = -FLT_MAX;
    }

    local float shared[GROUP_SIZE];

    max_val = group_max(shared, max_val);

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data[i] = exp(thread_data[i] - max_val);
            sum_val += thread_data[i];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sum_val = group_sum(shared, sum_val);

    for (int i = 0; i < items; i++)
        if (local_base + i < att_len)
            att[global_base + i] = thread_data[i] / sum_val;
}

kernel void softmax_general(
    global float *att,
    Tidx const seq_len,
    Tidx const att_len) {

    Tidx local_id = get_local_id(0),
         group_id = get_group_id(0),
         global_id = get_global_id(0),
         local_size = get_local_size(0);

    int items = (att_len + local_size - 1) / local_size;
    int local_base = local_id * items;
    int global_base = group_id * att_len + local_base;

    float thread_data, max_val = -FLT_MAX, sum_val = 0;

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data = (att_len + group_id % seq_len >= (local_base + i) + seq_len)
                              ? att[global_base + i]
                              : -FLT_MAX;

            if (max_val < thread_data)
                max_val = thread_data;
        }
    }

    local float shared[GROUP_SIZE];

    max_val = group_max(shared, max_val);

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data = (att_len + group_id % seq_len >= (local_base + i) + seq_len)
                              ? att[global_base + i]
                              : -FLT_MAX;
            thread_data = exp(thread_data - max_val);
            sum_val += thread_data;
            att[global_base + i] = (att_len + group_id % seq_len >= (local_base + i) + seq_len)
                                       ? thread_data
                                       : 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sum_val = group_sum(shared, sum_val);

    for (int i = 0; i < items; i++)
        if (local_base + i < att_len)
            att[global_base + i] /= sum_val;
}
