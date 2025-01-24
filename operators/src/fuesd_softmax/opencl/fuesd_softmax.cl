#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define BLOCK_SIZE 512
#define ITEMS 16
__kernel void softmax_padding(
    __global float *att,
    unsigned int seq_len,
    unsigned int att_len) {

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = group_id * att_len + local_id;
    int local_size = get_local_size(0);

    __local float localA[BLOCK_SIZE];
    float max_val, sum_val;
    float thread_data;
    if (local_id < att_len)
        thread_data = (att_len + (group_id % seq_len) >= local_id + seq_len) ? att[global_id] : -FLT_MAX;
    else
        thread_data = -FLT_MAX;

    localA[local_id] = thread_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            int partner = local_id + stride;
            if (partner < att_len) {
                localA[local_id] = fmax(localA[local_id], localA[partner]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    max_val = localA[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    thread_data = exp(thread_data - max_val);

    if (local_id >= att_len)
        thread_data = 0;
    localA[local_id] = thread_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            int partner = local_id + stride;
            if (partner < att_len) {
                localA[local_id] += localA[partner];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum_val = localA[0];
    thread_data = thread_data / sum_val;
    if (local_id < att_len) {
        att[global_id] = thread_data;
    }
}

__kernel void softmax_folding(
    __global float *att,
    unsigned int seq_len,
    unsigned int att_len) {

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    int items = (att_len + local_size - 1) / local_size;
    int local_base = local_id * items;
    int global_base = group_id * att_len + local_base;
    __local float localA[BLOCK_SIZE];
    float max_val, sum_val;
    float thread_data[ITEMS];

    max_val = -FLT_MAX;
    sum_val = 0;

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
    localA[local_id] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            localA[local_id] = fmax(localA[local_id], localA[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    max_val = localA[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data[i] = exp(thread_data[i] - max_val);
            sum_val += thread_data[i];
        }
    }
    localA[local_id] = sum_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            localA[local_id] += localA[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_val = localA[0];

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data[i] = thread_data[i] / sum_val;
            att[global_base + i] = thread_data[i];
        }
    }
}

__kernel void softmax_general(
    __global float *att,
    unsigned int seq_len,
    unsigned int att_len) {

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    int items = (att_len + local_size - 1) / local_size;
    int local_base = local_id * items;
    int global_base = group_id * att_len + local_base;
    __local float localA[BLOCK_SIZE];
    float max_val, sum_val;
    float thread_data;

    max_val = -FLT_MAX;
    sum_val = 0;

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data = (att_len + group_id % seq_len >= (local_base + i) + seq_len)
                              ? att[global_base + i]
                              : -FLT_MAX;

            if (max_val < thread_data)
                max_val = thread_data;
        }
    }
    localA[local_id] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            localA[local_id] = fmax(localA[local_id], localA[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    max_val = localA[0];
    barrier(CLK_LOCAL_MEM_FENCE);

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
    localA[local_id] = sum_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            localA[local_id] += localA[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_val = localA[0];

    for (int i = 0; i < items; i++) {
        if (local_base + i < att_len) {
            thread_data = att[global_base + i];
            thread_data = thread_data / sum_val;
            att[global_base + i] = thread_data;
        }
    }
}