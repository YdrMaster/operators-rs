#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENTION cl_khr_fp16
#define TILE_SIZE 256

typedef struct {
    unsigned int idx;
    float val;
} KVPair;
__kernel void argmax_step1(
    __global float *input,
    const int n) {
    //获取相关线程和块的索引
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    __local float local_max_value[TILE_SIZE];
    __local int local_max_index[TILE_SIZE];
    //加载到共享
    local_max_value[local_id] = (global_id < n) ? *(input + global_id) : -1;
    local_max_index[local_id] = (global_id < n) ? global_id : -1;
    barrier(CLK_LOCAL_MEM_FENCE);//加载后同步

    //规约求最大
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            if (local_max_value[local_id] < local_max_value[local_id + offset]) {
                local_max_value[local_id] = local_max_value[local_id + offset];
                local_max_index[local_id] = local_max_index[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        *(input + (global_id / 256)) = local_max_value[0];
        *(input + TILE_SIZE + (global_id / 256)) = local_max_index[0];
    }
}
__kernel void argmax_step2(
    __global float *input,
    __global KVPair *kvpair,
    const int n) {
    //获取相关线程和块的索引
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    __local float local_max_value[TILE_SIZE];
    __local int local_max_index[TILE_SIZE];
    //加载到共享
    local_max_value[local_id] = (global_id < n) ? *(input + global_id) : -1;
    local_max_index[local_id] = (global_id < n) ? *(input + TILE_SIZE + local_id) : -1;
    barrier(CLK_LOCAL_MEM_FENCE);//加载后同步

    //规约求最大
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            if (local_max_value[local_id] < local_max_value[local_id + offset]) {
                local_max_value[local_id] = local_max_value[local_id + offset];
                local_max_index[local_id] = local_max_index[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //存储最大值
    if (local_id == 0) {
        kvpair[0].val = local_max_value[0];
        kvpair[0].idx = local_max_index[0];
    }
}


// #define RADIX_BITS 4
// 将 float 转换为 uint
// inline uint float_to_uint(float f) {
//     uint i = as_uint(f);
//     if (i & 0x80000000) {
//         // 对于负数，将符号位翻转
//         i = ~i;
//     } else {
//         // 对于正数，将符号位置为最高位
//         i = i | 0x80000000;
//     }
//     return i;
// }

// // 将 uint 转换回 float
// inline float uint_to_float(uint i) {
//     if (i & 0x80000000) {
//         // 对于正数，恢复符号位
//         i = i & 0x7FFFFFFF;
//     } else {
//         // 对于负数，将符号位恢复
//         i = ~i;
//     }
//     return as_float(i);
// }

// __kernel void argmax(
//     __global float* input,
//     __global KVPair* kvpair,
//     const int n
// ){
//     //获取相关线程和块的索引
//     int global_id = get_global_id(0);
//     int local_id = get_local_id(0);
//     int group_id = get_group_id(0);
//     int local_size = get_local_size(0);
//     // int group_size = (n + local_size - 1) /local_size;
//     //使用共享内存加速规约
//     __local float local_max_value[TILE_SIZE];
//     __local int local_max_index[TILE_SIZE];
//     //加载到共享
//     local_max_value[local_id] = (global_id < n) ? *(input + global_id) : -1;
//     local_max_index[local_id] = (global_id < n) ? global_id : -1;
//     barrier(CLK_LOCAL_MEM_FENCE); //加载后同步

//     //规约求最大
//     for (int offset = local_size / 2; offset > 0; offset /= 2) {
//         if (local_id < offset) {
//             if (local_max_value[local_id] < local_max_value[local_id + offset]) {
//                 local_max_value[local_id] = local_max_value[local_id + offset];
//                 local_max_index[local_id] = local_max_index[local_id + offset];
//             }
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }
//     if(local_id == 0)
//     {
//         *(input + (global_id / 256)) = local_max_value[0];
//         *(input + TILE_SIZE + (global_id / 256)) = local_max_index[0];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);//这里的问题，这个只是块内同步，不知道别的人写完了没
//     if(global_id < 256)
//     {
//         local_max_value[local_id] = (local_id < 125) ? *(input + local_id) : -1;
//         local_max_index[local_id] = (local_id < 125) ? *(input + TILE_SIZE + local_id) : -1;
//         barrier(CLK_LOCAL_MEM_FENCE); //加载后同步
//         for (int offset = local_size / 2; offset > 0; offset /= 2) {
//             if (local_id < offset) {
//                 if (local_max_value[local_id] < local_max_value[local_id + offset]) {
//                     local_max_value[local_id] = local_max_value[local_id + offset];
//                     local_max_index[local_id] = local_max_index[local_id + offset];
//                 }
//             }
//             barrier(CLK_LOCAL_MEM_FENCE);
//         }
//         //存储最大值
//         if (local_id == 0) {
//             kvpair[0].val = local_max_value[0];
//             kvpair[0].idx = local_size;
//         }
//     }
// }

// __kernel void argmax(
//     __global const float* input,
//     __global int* max_index,
//     __global float* max_value,
//     const int n
// ){
//     //获取相关线程和块的索引
//     int global_idx = get_global_id(0);
//     int att_idx = get_local_id(0);
//     int token_idx = get_group_id(0);
//     int local_size = get_local_size(0);
//     //使用共享内存加速规约
//     __local float local_max_value[256];
//     __local int local_max_index[256];
//     //加载到共享
//     local_max_value[local_id] = (gid < n) ? input[gid] : -FLT_MAX;
//     local_max_index[local_id] = gid;
//     barrier(CLK_LOCAL_MEM_FENCE); //加载后同步

//     //规约求最大
//     for (int offset = group_size / 2; offset > 0; offset /= 2) {
//         if (local_id < offset) {
//             if (local_max_value[local_id] < local_max_value[local_id + offset]) {
//                 local_max_value[local_id] = local_max_value[local_id + offset];
//                 local_max_index[local_id] = local_max_index[local_id + offset];
//             }
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }
//     //存储最大值
//     if (local_id == 0) {
//         max_value[get_group_id(0)] = local_max_value[0];
//         max_index[get_group_id(0)] = local_max_index[0];
//     }
// }

// __kernel void RadixSort(
//     __global float *keys,       // 输入：待排序的键 (浮点数)
//     __global float *temp_keys,      // 临时存储空间 (浮点数)
//     __global uint *histogram,       // 输出：每个位段的直方图
//     const uint num_keys,            // 键的数量
//     const uint bit_shift,           // 当前处理的位段
//     ){       // 局部内存直方图  这里局部内存是不是应该放在内部申请;

//     //获取线程索引
//     int gid = get_global_id(0);
//     int lid = get_local_id(0);
//     int group_size = get_local_size(0);

//     //local  块内局部直方图处理
//     int num = 1 << RADIX_BITS;
//     __local int local_hist[num];
//     //每个线程处理对应的桶,为了充分利用计算资源,尽可能保证,num是group_size的整数倍
//     for(i = lid; i < num; i += group_size)
//         local_hist[i]=0;
//     barrier(CLK_LOCAL_MEM_FENCE);  // 同步，确保所有局部内存初始化完成
//     //计算当前线程处理的键
//     uint digit = 0;
//     if(gid < num_keys)
//     {
//         key = float_to_int(keys[gid]);
//         uint digit = (key >> bit_shift) & (num - 1);
//         atomic_inc(&local_hist[digit]);  //共享内存中直方图计数
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);  // 同步，确保所有直方图计算结束
//     //atomic_inc和 是原子操作保证数据一致性
//     //合并各块计算结果到全局
//     for(int  i = lid; i < num; i += group_size)
//     {
//         atomic_add(&histogram[i], local_hist[i]);
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);  // 同步，确保全局直方图更新结束

//     //前缀和
//     for(int offset = 1; offset < n; offset *=2)
//     {
//         if(gid >= offset && gid < n)
//         {
//             float temp = data[gid - offset];
//             data[gid] += temp;// todo :该语句执行之前是否需要同步操作
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }
//     //排序
//     if(gid < num_keys)
//     {
//         uint pos = atomic_add(&histogram[digit], -1);  //digit在上面已计算;
//         temp_keys[pos - 1] = key;
//     }

//     ////方式二，应该是比较低效
//     // // 计算全局直方图前缀和，决定每个工作组处理的数据区间
//     // uint offset = 0;
//     // for (int i = 0; i < group_id; i++) {
//     //     offset += histogram[i * num_bins + local_id];
//     // }

//     // // 使用直方图重排输入数据
//     // for (int i = global_id; i < num_elements; i += get_global_size(0)) {
//     //     uint key = input_keys[i];
//     //     uint bin = (key >> shift) & mask;
//     //     uint pos = offset + atomic_inc(&histogram[bin + group_id * num_bins]);
//     //     output_keys[pos] = key;
//     // }


// }

// __kernel void InclusiveSum (__global float* data, int n)
// {
//     //获取线程信息
//     int global_idx = get_global_id(0);           //线程在全局id

//     // //用平衡树规约求前缀和--方法1
//     // for(int offset = 1; offset < n; offset *= 2)
//     // {
//     //     float temp = data[global_idx];
//     //     if(global_idx >= offset && global_idx < n){
//     //         temp += data[global_idx-offset];
//     //         data[global_idx] = temp;
//     //     }
//     //     barrier(CLK_LOCAL_MEM_FENCE);
//     // }
//     //用平衡树规约求前缀和--方法2
//     for(int offset = 1; offset < n; offset *= 2)
//     {
//         if(global_idx >= offset && global_idx < n){
//             float temp = data[global_idx-offset];
//             data[global_idx] += temp;
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }

//     //加载数据到共享内存   ---前缀和没法用共享内存
//     // __local float local_data[256];
//     // if (gid < n) {
//     //     local_data[local_id] = data[gid];
//     // } else {
//     //     local_data[local_id] = 0.0f;
//     // }
//     // barrier(CLK_LOCAL_MEM_FENCE);               //同步

//     // //平衡树规约 求前缀和
//     // for(int offset = 1; offset < n; offset *= 2)
//     // {
//     //     float temp = data[global_idx];
//     //     barrier(CLK_LOCAL_MEM_FENCE);
//     //     if(global_idx >= offset){
//     //         temp += data[global_idx-offset];
//     //     }
//     // }
//     // for(int offset = 1; offset < n; offset *= 2)
//     // {
//     //     float temp = 0;
//     //     barrier(CLK_LOCAL_MEM_FENCE);
//     //     if(global_idx >= offset){
//     //         da
//     //     }
//     // }

// }

// __kernel void PartialSoftmax(
//     __global float* data,
//     int n,
//     float temprature
// ){
//     //获取索引
//     int global_idx = get_global_id(0);          //线程在全局id
//     int local_idx = get_local_id(0);            //工作组局部id
//     int local_size = get_local_size(0);         //工作组尺寸

//     //获取max
//     float max = data[0];

//     //加载到缓存 避免多次读写
//     __local float localA[256];    //等同工作组大小
//     __ local float  sum_val;
//     float threaddata = data[global_idx]

//     //减去最大值并计算指数
//     threaddata = exp((threaddata-max)/temprature);

//     //结果存入共享内存
//     localA[local_idx] = threaddata;

//     //规约求和
//     for(int stride = local_size / 2; stride > 0; stride /= 2)
//     {
//         if(att_idx < stride)
//         {
//             localA[att_idx] += localA[att_idx+stride];
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }
//     //获取结果
//     if (att_idx == 0) {
//             sum_val = localA[0];                // 获取最大值
//         }
//     barrier(CLK_LOCAL_MEM_FENCE);

//     //归一化
//     thread_data = thread_data/ sum_val;


//     //写入结果
//     att[global_idx] = thread_data;

// }

// typedef struct{
//         int key;
//         float value;
//     } KVPair;
// __kernel void RandomSample(
//     __global KVPair* kvpair,
//     __global float* sorted_val,
//     __global float* sorted_index,
//     size_t n;
//     float topp,
//     float topk,
//     float random){
//         //先根据topp topk 和random计算出实际的p值,1.就算topk和topp的最小范围(就是公共子集)=P(对应前缀和概率)
//         //2.然后乘以一个随机值(0,1)
//         //就是计算出来随机到哪个了,然后再把那个取出来,放到kvpair
//         //计算P值
//         topk = min(topk,n);
//         float p = random * min(topp * sorted_value[n-1],sorted[topk-1]);

//         //遍历排序后数组,找到随机到的P对应的值和索引
//         for(size_t i=0; i<n; i++)
//         {
//             if(sorted_value[i]>=p)
//             {
//                 kvpair->key = sorted_index[i];
//                 kvpair->value = sorted_value[i];
//                 return;
//             }
//         }
// }