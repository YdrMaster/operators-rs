#include <cub/block/block_reduce.cuh>

struct AttentionCausualMask {
    __forceinline__ __device__ bool
    operator()(int tok_id, int seq_len,
               int pos_id, int att_len) {
        //   tok_id ↓ |<---att_len--->|
        //          0 | * * ... *     |
        //          1 | * * ... * *   |
        //          2 | * * ... * * * |
        // seq_len: 3 |---------------|
        return att_len + tok_id >= pos_id + seq_len;
    }
};

template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
static __device__ void block_padding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const token_idx,
    unsigned int const seq_len) {

    auto att_idx = threadIdx.x, att_len = blockDim.x;
    auto thread_data = mask(token_idx, seq_len, att_idx, att_len)
                           ? float(att[att_idx])
                           : -__FLT_MAX__;

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_data, cub::Max(), att_len);
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        auto acc = block_op.Sum(thread_data = expf(thread_data - max), att_len);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    att[att_idx] = Tdata(thread_data * mean);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Tmask>
static __device__ void block_folding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const token_idx,
    unsigned int const seq_len,
    unsigned int const att_len) {

    auto local = (att_len + blockDim.x - 1) / blockDim.x;

    auto thread_offset = threadIdx.x * local;
    att += thread_offset;

    float thread_data[ITEMS_PER_THREAD], thread_max = -__FLT_MAX__, thread_sum = 0;
    for (unsigned int i = 0; i < local; ++i) {
        auto att_idx = thread_offset + i;
        thread_data[i] = att_idx < att_len && mask(token_idx, seq_len, att_idx, att_len)
                             ? float(att[i])
                             : -__FLT_MAX__;
        thread_max = cub::Max()(thread_max, thread_data[i]);
    }

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto block_op = BlockOp(temp_storage);

    __shared__ float max;
    {
        auto acc = block_op.Reduce(thread_max, cub::Max());
        if (threadIdx.x == 0) { max = acc; }
    }
    __syncthreads();

    __shared__ float mean;
    {
        for (unsigned int i = 0; i < local; ++i) {
            thread_data[i] = expf(thread_data[i] - max);
            thread_sum += thread_data[i];
        }
        auto acc = block_op.Sum(thread_sum);
        if (threadIdx.x == 0) { mean = fdividef(1, acc); }
    }
    __syncthreads();

    for (unsigned int i = 0; i < local; ++i) {
        if (auto att_idx = thread_offset + i; att_idx < att_len) {
            att[i] = Tdata(thread_data[i] * mean);
        }
    }
}

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata, class Tmask>
static __forceinline__ __device__ void padding(
    Tdata *__restrict__ att,
    Tmask mask,
    int const stride_z,
    int const stride_y,
    int const stride_x) {
    auto offset = blockIdx.x * stride_x + blockIdx.y * stride_y + blockIdx.z * stride_z,
         token_idx = blockIdx.x,
         seq_len = gridDim.x;
    block_padding<BLOCK_SIZE>(
        att + offset, mask, token_idx, seq_len);
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Tmask>
static __forceinline__ __device__ void folding(
    Tdata *__restrict__ att,
    Tmask mask,
    unsigned int const att_len,
    int const stride_z,
    int const stride_y,
    int const stride_x) {
    auto offset = blockIdx.x * stride_x + blockIdx.y * stride_y + blockIdx.z * stride_z,
         token_idx = blockIdx.x,
         seq_len = gridDim.x;
    block_folding<BLOCK_SIZE, ITEMS_PER_THREAD>(
        att + offset, mask, token_idx, seq_len, att_len);
}
