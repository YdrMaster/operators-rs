#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 8
#endif

#ifndef MASK
#define MASK causal_mask
#endif

typedef unsigned int Tidx;

bool causal_mask(Tidx tok_id, Tidx seq_len,
                 Tidx pos_id, Tidx att_len) {
    //   tok_id ↓ |<---att_len--->|
    //          0 | * * ... *     |
    //          1 | * * ... * *   |
    //          2 | * * ... * * * |
    // seq_len: 3 |---------------|
    return att_len + tok_id >= pos_id + seq_len;
}

kernel void softmax_register(
    global Tval *att_,
    Tidx const seq_len,
    Tidx const att_len,
    int const head_stride,
    int const tok_stride) {

    Tidx const
        head_idx = get_group_id(1),
        tok_id = get_group_id(0),
        l_idx = get_local_id(0),
        l_len = get_local_size(0);

    global Tval *att = att_ + head_idx * head_stride + tok_id * tok_stride;

    float
        data[ITEMS_THREAD],
        max_ = -FLT_MAX,
        sum_ = 0;

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        data[i] = causal_mask(tok_id, seq_len, idx, att_len) ? att[idx] : -FLT_MAX;
        max_ = fmax(max_, data[i]);
    }

    max_ = work_group_reduce_max(max_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        data[i] = exp(data[i] - max_);
        sum_ += data[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float const k = 1 / work_group_reduce_add(sum_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len)
        att[idx] = data[i] * k;
}

kernel void softmax_global(
    global Tval *att_,
    Tidx const seq_len,
    Tidx const att_len,
    int const head_stride,
    int const tok_stride) {

    Tidx const
        head_idx = get_group_id(1),
        tok_id = get_group_id(0),
        l_idx = get_local_id(0),
        l_len = get_local_size(0);

    global Tval *att = att_ + head_idx * head_stride + tok_id * tok_stride;

    float
        max_ = -FLT_MAX,
        sum_ = 0;

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        float const data = causal_mask(tok_id, seq_len, idx, att_len) ? att[idx] : -FLT_MAX;
        max_ = fmax(max_, data);
    }

    max_ = work_group_reduce_max(max_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        float const data = exp(att[idx] - max_);
        att[idx] = data;
        sum_ += data;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float const k = 1 / work_group_reduce_add(sum_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len)
        att[idx] *= k;
}
