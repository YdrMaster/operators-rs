#include <cub/device/device_reduce.cuh>

template<class T>
cudaError argmax(
    void *temp_storage, size_t *temp_storage_bytes,
    cub::KeyValuePair<uint64_t, T> *output, T const *data, unsigned int n,
    cudaStream_t stream) {
    return cub::DeviceReduce::ArgMax(
        temp_storage, *temp_storage_bytes,
        data, output, n,
        stream);
}
