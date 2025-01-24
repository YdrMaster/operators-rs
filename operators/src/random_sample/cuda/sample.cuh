#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>

template<class T>
cudaError arg_max_(
    cub::KeyValuePair<int, T> *kv_pair,
    T const *logits,
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    cudaStream_t stream) {
    return cub::DeviceReduce::ArgMax(
        workspace_ptr, workspace_len,
        logits, kv_pair, n,
        stream);
}

template<class T, class I>
cudaError radix_sort(
    void *workspace_ptr, size_t &workspace_len,
    T const *key_in, T *key_out,
    I const *val_in, I *val_out,
    int n,
    cudaStream_t stream) {
    return cub::DeviceRadixSort::SortPairsDescending(
        workspace_ptr, workspace_len,
        key_in, key_out,
        val_in, val_out,
        n,
        0, sizeof(T) * 8,
        stream);
}

template<class T>
cudaError inclusive_sum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(
        workspace_ptr, workspace_len,
        data, data, n,
        stream);
}

template<class T>
__global__ void partial_softmax_kernel(
    T *__restrict__ data, int n,
    float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < n) {
        float max = __ldg(data);
        data[i] = (T) expf(((float) data[i] - max) / temperature);
    }
}

template<class T>
__global__ void set_softmax_max_kernel(
    T *__restrict__ data) {
    *data = 1;
}

template<class T, class I>
__global__ void random_sample_kernel(
    cub::KeyValuePair<int, T> *__restrict__ kv_pair,
    T const *__restrict__ sorted,
    I const *__restrict__ indices_out,
    size_t n,
    float random, float topp, size_t topk) {
    topk = cub::Min()(topk, n);
    auto p = (T) (random * cub::Min()(topp * (float) sorted[n - 1], (float) sorted[topk - 1]));
    for (size_t i = 0;; ++i) {
        if ((sorted[i]) >= p) {
            kv_pair->key = indices_out[i];
            kv_pair->value = sorted[i];
            return;
        }
    }
}

#define CHECK(statement)                                                                        \
    {                                                                                           \
        auto error = (statement);                                                               \
        if (error != cudaSuccess) {                                                             \
            printf("Error: %s (%d) at \"%s\"\n", cudaGetErrorString(error), error, #statement); \
            return error;                                                                       \
        }                                                                                       \
    }

constexpr size_t align(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

template<class T>
cudaError calculate_workspace_size(
    size_t *argmax,
    size_t *random_sample,
    size_t n_) {
    auto const n = static_cast<int>(n_);

    CHECK(arg_max_<T>(
        nullptr, nullptr, n,
        nullptr, *argmax,
        nullptr))

    size_t size_radix_sort;
    CHECK((radix_sort<T, unsigned int>(
        nullptr, size_radix_sort,
        nullptr, nullptr,
        nullptr, nullptr,
        n,
        nullptr)))

    size_t size_inclusive_sum;
    CHECK(inclusive_sum<T>(
        nullptr, size_inclusive_sum,
        nullptr, n,
        nullptr))

    size_t size_random = 0;
    size_random += sizeof(T) * n;           // sorted
    size_random = align(size_random, 256);  //
    size_random += sizeof(unsigned int) * n;// indices_out
    size_random = align(size_random, 256);  //
    size_random += cub::Max()(size_radix_sort, size_inclusive_sum);
    *random_sample = size_random;

    return cudaGetLastError();
}

template<class T>
cudaError arg_max(
    cub::KeyValuePair<int, T> *kv_pair,
    T const *logits,
    size_t n,

    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream) {
    return arg_max_(
        kv_pair,
        logits,
        n,
        workspace_ptr,
        workspace_len,
        stream);
}

template<class T>
cudaError random_sample(
    cub::KeyValuePair<int, T> *kv_pair,
    T const *logits,
    unsigned int const *indices,
    size_t n,

    float random,
    float temperature,
    float topp,
    size_t topk,

    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream) {

    auto workspace = reinterpret_cast<size_t>(workspace_ptr);
    auto workspace_end = workspace + workspace_len;

    auto sorted = reinterpret_cast<T *>(workspace);
    workspace += sizeof(T) * n;
    workspace = align(workspace, 256);

    auto indices_out = reinterpret_cast<unsigned int *>(workspace);
    workspace += sizeof(unsigned int) * n;
    workspace = align(workspace, 256);

    workspace_ptr = reinterpret_cast<void *>(workspace);
    workspace_len = workspace_end - workspace;

    // sort
    CHECK(radix_sort(
        workspace_ptr, workspace_len,
        logits, sorted,
        indices, indices_out,
        n,
        stream));
    // softmax
    auto block = cub::Min()((size_t) 1024, n);
    auto grid = (n + block - 1) / block;
    partial_softmax_kernel<<<grid, block, 0, stream>>>(sorted, n, temperature);
    set_softmax_max_kernel<<<1, 1, 0, stream>>>(sorted);
    // sum
    CHECK(inclusive_sum(
        workspace_ptr, workspace_len,
        sorted, n,
        stream));
    // sample
    random_sample_kernel<<<1, 1, 0, stream>>>(
        kv_pair,
        sorted, indices_out, n,
        random, topp, topk);
    return cudaGetLastError();
}
