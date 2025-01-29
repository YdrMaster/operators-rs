#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENTION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

// assert: GROUP_SIZE is power of 2
#ifndef GROUP_SIZE
#define GROUP_SIZE 1024
#endif

typedef unsigned int Tidx;

typedef struct {
    Tidx idx;
    Tval val;
} KVPair;

void argmax_local(local KVPair *kv_pairs, KVPair reg) {
    Tidx const l_idx = get_local_id(0);

    kv_pairs[l_idx] = reg;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (Tidx stride = GROUP_SIZE >> 1; stride > 0; stride >>= 1) {
        if (l_idx < stride) {
            local KVPair
                *a = kv_pairs + l_idx,
                *b = kv_pairs + l_idx + stride;
            if (b->val > a->val) *a = *b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void argmax_build_pairs(
    global Tval const *input,
    global KVPair *output,
    Tidx const n,
    Tval init) {

    Tidx const
        g_idx = get_global_id(0),
        g_len = get_global_size(0),
        l_idx = get_local_id(0);

    // register: 每个线程可能处理多个数据，汇总到寄存器中
    // NOTICE 为保证线程利用率，每个线程应该处理至少 2 个数据
    KVPair reg = {-1, init};
    for (Tidx i = g_idx; i < n; i += g_len) {
        Tval const val = input[i];
        if (val > reg.val) reg = (KVPair) {i, val};
    }

    // local memory: 每个工作组在工作组内存中实现规约
    local KVPair kv_pairs[GROUP_SIZE];
    argmax_local(kv_pairs, reg);

    // 最终结果写回 global
    if (l_idx == 0) output[g_idx / GROUP_SIZE] = *kv_pairs;
}

kernel void argmax_reduce(
    global KVPair const *pairs,
    global KVPair *output,
    Tidx const n,
    Tval init) {

    Tidx const
        g_idx = get_global_id(0),
        g_len = get_global_size(0),
        l_idx = get_local_id(0);

    // register: 每个线程可能处理多个数据，汇总到寄存器中
    // NOTICE 为保证线程利用率，每个线程应该处理至少 2 个数据
    KVPair reg = {-1, init};
    for (Tidx i = g_idx; i < n; i += g_len) {
        KVPair const pair = pairs[i];
        if (pair.val > reg.val) reg = pair;
    }

    // local memory: 每个工作组在工作组内存中实现规约
    local KVPair kv_pairs[GROUP_SIZE];
    argmax_local(kv_pairs, reg);

    // 最终结果写回 global
    if (l_idx == 0) *output = *kv_pairs;
}
