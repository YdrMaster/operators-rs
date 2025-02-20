#include <cstdint>// 为int32_t添加定义

template<class Tmem>
static __device__ void rearrange(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const ncols) {

    auto row = blockIdx.y,
         col = blockIdx.x * blockDim.y + threadIdx.y;
    if (col >= ncols) return;

    auto thread = threadIdx.x,
         warp_size = blockDim.x;
    auto i = (row * rsa + col * csa) * warp_size + thread;
    auto j = (row * rsb + col * csb) * warp_size + thread;

    reinterpret_cast<Tmem *>(dst)[i] = reinterpret_cast<Tmem const *>(src)[j];
}

// 共享内存版本的rearrange kernel
// 尽量保证：rsb 连续，也就是输入的r是连续的，row读取连续，blockIdx.x对应row
// 尽量保证：csa 连续，也就是输出的c是连续的，col写入连续，blockIdx.y对应col
// 读入时，每个warp读取sub_size_x个元素，共有sub_size_y个warp这么干
// 写入时，每个warp写入sub_size_y个元素，共有sub_size_x个warp这么干
template<class Tmem>
static __device__ void rearrange_shared(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const nrows,
    unsigned int const ncols,
    unsigned int const sub_size_x,
    unsigned int const sub_size_y) {
    extern __shared__ char smem[];
    Tmem *shared = reinterpret_cast<Tmem *>(smem);

    // 计算线程和 warp 的索引
    const int warp_thread_idx = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int row_base = blockIdx.x * sub_size_x;
    const int col_base = blockIdx.y * sub_size_y;

    // 读取阶段：每个 warp 中只有前 sub_size_x 个线程工作
    if (warp_thread_idx < sub_size_x && warp_idx < sub_size_y) {
        const int row = row_base + warp_thread_idx;
        const int col = col_base + warp_idx;
        if (row < nrows && col < ncols) {
            const int src_idx = row * rsb + col * csb;
            const int smem_idx = warp_thread_idx * sub_size_y + warp_idx;
            shared[smem_idx] = reinterpret_cast<const Tmem *>(src)[src_idx];
        }
    }

    __syncthreads();

    // 写入阶段：每个 warp 中只有前 sub_size_y 个线程工作
    if (warp_thread_idx < sub_size_y && warp_idx < sub_size_x) {
        const int col = col_base + warp_thread_idx;
        const int row = row_base + warp_idx;
        if (row < nrows && col < ncols) {
            const int dst_idx = row * rsa + col * csa;
            const int smem_idx = warp_idx * sub_size_y + warp_thread_idx;
            reinterpret_cast<Tmem *>(dst)[dst_idx] = shared[smem_idx];
        }
    }
}


template<class Tmem>
static __device__ void rearrange2(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const nrows,
    unsigned int const ncols,
    unsigned int const sub_size_x,
    unsigned int const sub_size_y) {
    // 计算线程和 warp 的索引
    const int warp_thread_idx = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int row_base = blockIdx.x * sub_size_x;
    const int col_base = blockIdx.y * sub_size_y;

    // 读取阶段：每个 warp 中只有前 sub_size_x 个线程工作
    if (warp_thread_idx < sub_size_x && warp_idx < sub_size_y) {
        const int row = row_base + warp_thread_idx;
        const int col = col_base + warp_idx;
        if (row < nrows && col < ncols) {
            const int src_idx = row * rsb + col * csb;
            const int dst_idx = row * rsa + col * csa;
            reinterpret_cast<Tmem *>(dst)[dst_idx] = reinterpret_cast<Tmem const *>(src)[src_idx];
        }
    }
}

template<class Tmem>
static __device__ void rearrange_large_unit(
    void *__restrict__ dst,
    int const rsa,
    int const csa,
    void const *__restrict__ src,
    int const rsb,
    int const csb,
    unsigned int const nrows,
    unsigned int const ncols,
    unsigned int const sub_size_x,
    unsigned int const sub_size_y,
    unsigned int const unit_size) {
    // 计算线程和 warp 的索引
    const int warp_thread_idx = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int row_base = blockIdx.x * sub_size_x;
    const int col_base = blockIdx.y * sub_size_y;

    // 计算每个线程需要处理的元素数量
    const int elements_per_thread = unit_size / sizeof(Tmem);

    // 读取阶段：每个 warp 中只有前 sub_size_x 个线程工作
    if (warp_thread_idx < sub_size_x && warp_idx < sub_size_y) {
        const int row = row_base + warp_thread_idx;
        const int col = col_base + warp_idx;
        if (row < nrows && col < ncols) {
            const int src_base_idx = row * rsb + col * csb;
            const int dst_base_idx = row * rsa + col * csa;

            // 使用循环处理多个元素
            for (int i = 0; i < elements_per_thread; i++) {
                reinterpret_cast<Tmem *>(dst)[dst_base_idx * elements_per_thread + i] =
                    reinterpret_cast<Tmem const *>(src)[src_base_idx * elements_per_thread + i];
            }
        }
    }
}

template<class Tmem>
static __device__ void rearrange_direct_copy(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const total_elements,
    unsigned int const unit_size = sizeof(Tmem)) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_thread = unit_size / sizeof(Tmem);

    if (tid < total_elements) {
        if (elements_per_thread == 1) {
            // 对于小于等于32字节的unit，直接拷贝
            reinterpret_cast<Tmem *>(dst)[tid] = reinterpret_cast<Tmem const *>(src)[tid];
        } else {
            // 对于大于32字节的unit，使用循环处理
            const int base_idx = tid * elements_per_thread;
            for (int i = 0; i < elements_per_thread; i++) {
                reinterpret_cast<Tmem *>(dst)[base_idx + i] = reinterpret_cast<Tmem const *>(src)[base_idx + i];
            }
        }
    }
}


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
                shared_constrains1_grid_idx_multiple = idx;
            }
            if (i == constrains2.a[0]) {
                shared_constrains2_grid_idx_multiple = idx;
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


template<class Tmem>
static __device__ void rearrange_1_blank(
    void *__restrict__ dst,
    void const *__restrict__ src,
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> block_len,       // 各维度的长度
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_block_stride,// 源tensor在各维度上的步长(bytes)
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,// 目标tensor在各维度上的步长(bytes)
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> grid_len,        // 各维度的长度
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_grid_stride, // 源tensor在各维度上的步长(bytes)
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride, // 目标tensor在各维度上的步长(bytes)
    unsigned int const unit_size                         // 每个元素的字节数
) {

    // 打印所有参数
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("size of int: %lu\n", sizeof(int));// 使用%lu代替%d来匹配unsigned long
        printf("参数信息:\n");
        printf("block_len: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", block_len.a[i]);
        }
        printf("\n");

        printf("src_block_stride: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", src_block_stride.a[i]);
        }
        printf("\n");

        printf("dst_block_stride: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", dst_block_stride.a[i]);
        }
        printf("\n");

        printf("grid_len: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", grid_len.a[i]);
        }
        printf("\n");

        printf("src_grid_stride: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", src_grid_stride.a[i]);
        }
        printf("\n");

        printf("dst_grid_stride: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", dst_grid_stride.a[i]);
        }
        printf("\n");

        printf("unit_size: %d\n", unit_size);
    }
}

// 表示维度切分信息
struct DimSplit {
    int outer_len;    // 外层循环次数
    int inner_len;    // 内层每次处理的长度
    int remainder_len;// 最后一次需要处理的剩余长度(可能小于inner_len)
    int total_len;    // 原始维度的总长度
};

struct SplitArrayStruct {
    DimSplit splits[ARRAY_SIZE];
};

template<class Tmem>
static __device__ void rearrange_2(
    void *__restrict__ dst,
    void const *__restrict__ src,
    SplitArrayStruct dim_splits,                         // 各维度的切分信息
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> src_block_stride,// 源tensor在各维度上的步长
    ArrayStruct<ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,// 目标tensor在各维度上的步长
    unsigned int const unit_size                         // 每个元素的字节数
) {
    // 计算当前block和thread在各维度上的内外层索引
    int outer_idx[ARRAY_SIZE];
    int inner_idx[ARRAY_SIZE];

    // 解码blockIdx.x得到外层索引
    int remaining = blockIdx.x;
#pragma unroll
    for (int i = ARRAY_SIZE - 1; i >= 0; i--) {
        outer_idx[i] = remaining % dim_splits.splits[i].outer_len;
        remaining /= dim_splits.splits[i].outer_len;
    }

    // 解码threadIdx.x得到内层索引
    remaining = threadIdx.x;
#pragma unroll
    for (int i = ARRAY_SIZE - 1; i >= 0; i--) {
        inner_idx[i] = remaining % dim_splits.splits[i].inner_len;
        remaining /= dim_splits.splits[i].inner_len;
    }

    // 计算实际访问的偏移
    int src_offset = 0;
    int dst_offset = 0;

#pragma unroll
    for (int i = 0; i < ARRAY_SIZE; i++) {
        // 计算当前维度的实际索引
        int actual_idx = outer_idx[i] * dim_splits.splits[i].inner_len + inner_idx[i];

        // 检查是否是最后一次外层循环，需要处理剩余部分
        if (outer_idx[i] == dim_splits.splits[i].outer_len - 1 &&
            actual_idx >= dim_splits.splits[i].total_len) {
            // 超出范围的线程不执行
            return;
        }

        src_offset += actual_idx * src_block_stride.a[i];
        dst_offset += actual_idx * dst_block_stride.a[i];
    }

    // 执行数据拷贝
    const int elements_per_thread = unit_size / sizeof(Tmem);
    // 将字节偏移转换为元素偏移
    const int src_elem_offset = src_offset / sizeof(Tmem);
    const int dst_elem_offset = dst_offset / sizeof(Tmem);

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
