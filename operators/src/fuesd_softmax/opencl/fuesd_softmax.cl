#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define BLOCK_SIZE 512
#define ITEM 16
__kernel void softmax_padding(
    __global float *att,
    unsigned int seq_len,
    unsigned int att_len) {

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    __local float localA[BLOCK_SIZE];
    float max_val, sum_val;
    float thread_data = (att_len + group_id % seq_len >= local_id + seq_len) ? att[global_id] : -FLT_MAX;

    localA[local_id] = thread_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = (local_size + 1) / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            int partner = local_id + stride;
            if (partner < local_size) {
                localA[local_id] = fmax(localA[local_id], localA[partner]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    max_val = localA[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    thread_data = exp(thread_data - max_val);
    localA[local_id] = thread_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = (local_size + 1) / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            int partner = local_id + stride;
            if (partner < local_size) {
                localA[local_id] += localA[partner];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum_val = localA[0];
    thread_data = thread_data / sum_val;
    att[global_id] = thread_data;
}