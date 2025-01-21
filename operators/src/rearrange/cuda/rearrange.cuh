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
    unsigned int const sub_size_y
) {
    extern __shared__ char smem[];
    Tmem* shared = reinterpret_cast<Tmem*>(smem);
    
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
            shared[smem_idx] = reinterpret_cast<const Tmem*>(src)[src_idx];
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
            reinterpret_cast<Tmem*>(dst)[dst_idx] = shared[smem_idx];
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
    unsigned int const sub_size_y
) {
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
    unsigned int const unit_size
) {
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
    unsigned int const unit_size = sizeof(Tmem)
) {
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

