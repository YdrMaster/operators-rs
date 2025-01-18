#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

struct SumPair {
    float average;
    float variance;

    __device__ SumPair operator+(const SumPair &other) const {
        return SumPair{this->average + other.average, this->variance + other.variance};
    }
};
template<unsigned int BLOCK_SIZE, class Ta, class Tw>
static __device__ void padding(
    Ta *__restrict__ y_,
    int const stride_y,
    Ta const *__restrict__ x_,
    int const stride_x,
    Tw const *__restrict__ s_,
    Tw const *__restrict__ b_,
    float const epsilon) {
    auto y = y_ + blockIdx.x * stride_y + threadIdx.x;
    float const
        x = x_[blockIdx.x * stride_x + threadIdx.x],
        s = s_[threadIdx.x],
        b = b_[threadIdx.x];

    using BlockOp = cub::BlockReduce<SumPair, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storge;
    SumPair tmp = {x, x * x};
    SumPair sum_pair = BlockOp(temp_storge).Reduce(tmp, cub::Sum());
    __shared__ float average, variance;
    if (threadIdx.x == 0) {
        average = sum_pair.average / float(BLOCK_SIZE);
        variance = __frcp_rn(sqrtf(sum_pair.variance / float(BLOCK_SIZE) - powf(average, 2.0)) + epsilon);
    }
    __syncthreads();

    *y = Ta((x - average) * variance * s + b);
}

template<unsigned int BLOCK_SIZE, unsigned int NUM_ITEMS_THREAD, class Tw, class Ta>
static __device__ void folding(
    Ta *__restrict__ y_,
    int const stride_y,
    Ta const *__restrict__ x_,
    int const stride_x,
    Tw const *__restrict__ s_,
    Tw const *__restrict__ b_,
    float const epsilon,
    unsigned int const items_size) {
    y_ += blockIdx.x * stride_y;
    x_ += blockIdx.x * stride_x;

    float data[NUM_ITEMS_THREAD], scale[NUM_ITEMS_THREAD], bias[NUM_ITEMS_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x_, data, items_size, 0.f);
        BlockOp(temp_storage).Load(s_, scale, items_size, 0.f);
        BlockOp(temp_storage).Load(b_, bias, items_size, 0.f);
    }

    float sum_average = 0, sum_variance = 0;
#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i) {
        sum_average += data[i];
        sum_variance += data[i] * data[i];
    }

    SumPair tmp_sum = {sum_average, sum_variance};
    using BlockOp = cub::BlockReduce<SumPair, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storge;
    SumPair sum_pair = BlockOp(temp_storge).Reduce(tmp_sum, cub::Sum());

    __shared__ float average, variance;
    if (threadIdx.x == 0) {
        average = sum_pair.average / float(items_size);
        variance = __frcp_rn(sqrtf(sum_pair.variance / float(items_size) - powf(average, 2.0)) + epsilon);
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i) {
        data[i] = (data[i] - average) * variance * scale[i] + bias[i];
    }

    {
        using BlockOp = cub::BlockStore<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Store(y_, data, items_size);
    }
}
