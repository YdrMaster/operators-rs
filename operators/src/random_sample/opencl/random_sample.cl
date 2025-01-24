#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENTION cl_khr_fp16 : enable
#define TILE_SIZE 256

typedef struct {
    unsigned int idx;
    float val;
} KVPair;
__kernel void argmax_step1(
    __global float *input,
    const int n) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    __local float local_max_value[TILE_SIZE];
    __local int local_max_index[TILE_SIZE];

    local_max_value[local_id] = (global_id < n) ? *(input + global_id) : -1;
    local_max_index[local_id] = (global_id < n) ? global_id : -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            if (local_max_value[local_id] < local_max_value[local_id + offset]) {
                local_max_value[local_id] = local_max_value[local_id + offset];
                local_max_index[local_id] = local_max_index[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        *(input + (global_id / 256)) = local_max_value[0];
        *(input + TILE_SIZE + (global_id / 256)) = local_max_index[0];
    }
}
__kernel void argmax_step2(
    __global float *input,
    __global KVPair *kvpair,
    const int n) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    __local float local_max_value[TILE_SIZE];
    __local int local_max_index[TILE_SIZE];

    local_max_value[local_id] = (global_id < n) ? *(input + global_id) : -1;
    local_max_index[local_id] = (global_id < n) ? *(input + TILE_SIZE + local_id) : -1;
    barrier(CLK_LOCAL_MEM_FENCE);


    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            if (local_max_value[local_id] < local_max_value[local_id + offset]) {
                local_max_value[local_id] = local_max_value[local_id + offset];
                local_max_index[local_id] = local_max_index[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        kvpair[0].val = local_max_value[0];
        kvpair[0].idx = local_max_index[0];
    }
}