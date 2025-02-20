#include <cstdint>// 为int32_t添加定义


#ifndef ARRAY_SIZE
#define ARRAY_SIZE 7
#endif

#ifndef ARRAY_TYPE
#define ARRAY_TYPE int// 使用int替代int32_t
#endif

template<int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};


template<class Tmem, int ArrSize, typename ArrayType>
static __device__ void rearrange_1(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,                    // block_len 各元素的乘积
    const ArrayStruct<4, ArrayType> constrains1,           // 切分维度的约束条件1，, 各个元素分别代表：[grid_idx, block_idx, grid 的stride相对于block的倍数，总的len限制]
    const ArrayStruct<4, ArrayType> constrains2,           // 切分维度的约束条件2
    const ArrayStruct<ArrSize, ArrayType> block_len,       // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_block_stride,// 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_block_stride,// 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> grid_len,        // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_grid_stride, // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_grid_stride, // 目标tensor在各维度上的步长(bytes)
    unsigned int const unit_size                           // 每个元素的字节数
) {

    int remaining = threadIdx.x;
    if (remaining >= block_len_total) {
        return;
    }

    // 声明共享内存
    __shared__ int shared_src_offset;
    __shared__ int shared_dst_offset;

    __shared__ int shared_constrains1_grid_idx_multiple;
    __shared__ int shared_constrains2_grid_idx_multiple;

    if (threadIdx.x == 0) {// 只让0号线程计算
        // 计算当前block处理的数据在src和dst中的基础偏移(bytes)
        int src_offset = 0;
        int dst_offset = 0;
        int remaining = blockIdx.x;
#pragma unroll
        for (int i = ARRAY_SIZE - 1; i >= 0; i--) {
            int idx = remaining % grid_len.a[i];
            remaining /= grid_len.a[i];
            src_offset += idx * src_grid_stride.a[i];
            dst_offset += idx * dst_grid_stride.a[i];

            if (i == constrains1.a[0]) {
                shared_constrains1_grid_idx_multiple = idx * constrains1.a[2];
            }
            if (i == constrains2.a[0]) {
                shared_constrains2_grid_idx_multiple = idx * constrains2.a[2];
            }

            // 将结果存入共享内存
            shared_src_offset = src_offset;
            shared_dst_offset = dst_offset;
        }
    }

    // 确保所有线程都能看到共享内存中的值
    __syncthreads();

    // 所有线程直接使用计算好的偏移值
    int src_offset = shared_src_offset;
    int dst_offset = shared_dst_offset;

    int constrains1_grid_idx_multiple = shared_constrains1_grid_idx_multiple;
    int constrains2_grid_idx_multiple = shared_constrains2_grid_idx_multiple;

    for (int i = ARRAY_SIZE - 1; i > 0; i--) {
        if (block_len.a[i] > 1) {
            int idx = remaining % block_len.a[i];
            remaining /= block_len.a[i];
            // 计算偏移量
            src_offset += idx * src_block_stride.a[i];
            dst_offset += idx * dst_block_stride.a[i];

            if (constrains1.a[3] != 0 && i == constrains1.a[1]) {
                if (constrains1_grid_idx_multiple + idx >= constrains1.a[3]) {
                    return;
                }
            }

            if (constrains2.a[3] != 0 && i == constrains2.a[1]) {
                if (constrains2_grid_idx_multiple + idx >= constrains2.a[3]) {
                    return;
                }
            }
        }
    }

    // 单独处理第一个维度
    if (remaining >= block_len.a[0]) {
        return;
    }
    src_offset += remaining * src_block_stride.a[0];
    dst_offset += remaining * dst_block_stride.a[0];

    if (constrains1.a[3] != 0 && 0 == constrains1.a[1]) {
        if (constrains1_grid_idx_multiple + remaining >= constrains1.a[3]) {
            return;
        }
    }

    if (constrains2.a[3] != 0 && 0 == constrains2.a[1]) {
        if (constrains2_grid_idx_multiple + remaining >= constrains2.a[3]) {
            return;
        }
    }

    // 执行数据拷贝，注意offset已经是字节偏移
    const int elements_per_thread = unit_size / sizeof(Tmem);
    if (elements_per_thread == 1) {
        *reinterpret_cast<Tmem *>(reinterpret_cast<char *>(dst) + dst_offset) =
            *reinterpret_cast<const Tmem *>(reinterpret_cast<const char *>(src) + src_offset);
    } else {
        for (int i = 0; i < elements_per_thread; i++) {
            reinterpret_cast<Tmem *>(reinterpret_cast<char *>(dst) + dst_offset)[i] =
                reinterpret_cast<const Tmem *>(reinterpret_cast<const char *>(src) + src_offset)[i];
        }
    }
}
