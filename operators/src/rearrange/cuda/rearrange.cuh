
#ifndef ARRAY_SIZE
#define ARRAY_SIZE 5
#endif

#ifndef ARRAY_TYPE
#define ARRAY_TYPE int
#endif

template<int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};

// 各个元素分别代表：[grid_idx, block_idx, grid 的stride相对于block的倍数，总的len限制]
template<typename ElementType>
struct Constrains {
    ElementType grid_idx;
    ElementType block_idx;
    ElementType grid_div_block;
    ElementType total_len;
};

// 主要的重排序内核模板
template<typename Tmem, int ConstrainNum>
__forceinline__ __device__ void rearrange_kernel(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,                        // block_len 各元素的乘积
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> block_len,       // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_block_stride,// 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,// 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> grid_len,        // 各维度的长度
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_grid_stride, // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride  // 目标tensor在各维度上的步长(bytes)
#if CONSTRAIN_NUM > 0
    ,
    const ArrayStruct<CONSTRAIN_NUM, Constrains<ARRAY_TYPE>> constrains// 切分维度的约束条件数组
#endif
) {
    int remaining = threadIdx.x;
    if (remaining >= block_len_total) {
        return;
    }

    // 声明共享内存
    __shared__ int shared_src_offset;
    __shared__ int shared_dst_offset;
#if CONSTRAIN_NUM > 0
    __shared__ int shared_constrains_grid_idx_multiple[CONSTRAIN_NUM];
#endif

    if (threadIdx.x == 0) {// 只让0号线程计算
        // 计算当前block处理的数据在src和dst中的基础偏移(bytes)
        int src_offset = 0;
        int dst_offset = 0;
#if CONSTRAIN_NUM > 0
        int constrains_grid_idx_multiple[CONSTRAIN_NUM] = {0};
#endif
        int remaining = blockIdx.x;

        for (int i = ARRAY_SIZE - 1; i >= 0; i--) {
            int idx = remaining % grid_len.a[i];
            remaining /= grid_len.a[i];
            src_offset += idx * src_grid_stride.a[i];
            dst_offset += idx * dst_grid_stride.a[i];
#if CONSTRAIN_NUM > 0
            for (int j = 0; j < CONSTRAIN_NUM; j++) {
                if (i == constrains.a[j].grid_idx) {
                    constrains_grid_idx_multiple[j] = idx * constrains.a[j].grid_div_block;
                }
            }
#endif
        }

        // 将结果存入共享内存
        shared_src_offset = src_offset;
        shared_dst_offset = dst_offset;
#if CONSTRAIN_NUM > 0
        for (int j = 0; j < CONSTRAIN_NUM; j++) {
            shared_constrains_grid_idx_multiple[j] = constrains_grid_idx_multiple[j];
        }
#endif
    }

    // 确保所有线程都能看到共享内存中的值
    __syncthreads();

    // 所有线程直接使用计算好的偏移值
    int src_offset = shared_src_offset;
    int dst_offset = shared_dst_offset;
#if CONSTRAIN_NUM > 0
    int constrains_grid_idx_multiple[CONSTRAIN_NUM];
    for (int j = 0; j < CONSTRAIN_NUM; j++) {
        constrains_grid_idx_multiple[j] = shared_constrains_grid_idx_multiple[j];
    }
#endif

    for (int i = ARRAY_SIZE - 1; i > 0; i--) {
        if (block_len.a[i] > 1) {
            int idx = remaining % block_len.a[i];
            remaining /= block_len.a[i];
            // 计算偏移量
            src_offset += idx * src_block_stride.a[i];
            dst_offset += idx * dst_block_stride.a[i];
#if CONSTRAIN_NUM > 0
            for (int j = 0; j < CONSTRAIN_NUM; j++) {
                if (constrains.a[j].total_len != 0 && i == constrains.a[j].block_idx) {
                    if (constrains_grid_idx_multiple[j] + idx >= constrains.a[j].total_len) {
                        return;
                    }
                }
            }
#endif
        }
    }

    src_offset += remaining * src_block_stride.a[0];
    dst_offset += remaining * dst_block_stride.a[0];
#if CONSTRAIN_NUM > 0
    for (int j = 0; j < CONSTRAIN_NUM; j++) {
        if (constrains.a[j].total_len != 0 && 0 == constrains.a[j].block_idx) {
            if (constrains_grid_idx_multiple[j] + remaining >= constrains.a[j].total_len) {
                return;
            }
        }
    }
#endif

    // 执行数据拷贝，注意offset已经是字节偏移
    *reinterpret_cast<Tmem *>(reinterpret_cast<char *>(dst) + dst_offset) =
        *reinterpret_cast<const Tmem *>(reinterpret_cast<const char *>(src) + src_offset);
}
