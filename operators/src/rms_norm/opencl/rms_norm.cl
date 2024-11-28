#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define BLOCK_SIZE 512
#define TILE_SIZE 16

__kernel void rms_norm_padding(
    __global float *y,
    const int y_stride,
    __global const float *x,
    const int x_stride,
    __global const float *w,
    const float epsilon) {

    //获取线程和块的索引
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    //计算输入输出指针
    int idx_x = group_id * x_stride + local_id;
    int idx_y = group_id * y_stride + local_id;

    //读取数据
    float val_x = x[idx_x];
    float val_w = w[local_id];

    //申请内存 求平方
    __local float squared[BLOCK_SIZE];
    squared[local_id] = val_x * val_x;
    barrier(CLK_LOCAL_MEM_FENCE);// 确保所有线程完成写入局部内存

    //规约求和(todo：处理local%2==1的情况)
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            squared[local_id] += squared[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);// 确保上一轮结束
    }
    //各块内线程规约求和   work_group_reduce_add:应该是opencl自带的规约函数,但是移动端GPU不支持

    //求均方根
    __local float rms;
    if (local_id == 0) {
        rms = native_rsqrt(squared[0] / local_size + epsilon);
    }
    barrier(CLK_LOCAL_MEM_FENCE);// 共享内存同步

    //计算最终结果并存储
    y[idx_y] = rms * val_x * val_w;
}

//note:仅支持d = 2^n ,如需支持其它尺寸，使用fused softmax的规约
__kernel void rms_norm_folding(
    __global float *y,
    const int y_stride,
    __global const float *x,
    const int x_stride,
    __global const float *w,
    const float epsilon,
    const int d) {

    //获取线程和块的索引
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    //每个工作项实际处理的个数
    int items = (d + local_size - 1) / local_size;

    //计算输入输出指针
    int idx_x = group_id * x_stride + local_id * items;
    int idx_y = group_id * y_stride + local_id * items;
    int idx_w = local_id * items;

    //读取数据
    float val_x[TILE_SIZE];
    float val_w[TILE_SIZE];
    for (int i = 0; i < items; i++) {
        val_x[i] = (local_id * items + i < d) ? x[idx_x + i] : 0.0f;
        val_w[i] = (local_id * items + i < d) ? w[idx_w + i] : 0.0f;
    }

    //各线程求局部平方和
    float squared = 0.0f;
    for (int i = 0; i < items; i++) {
        squared += val_x[i] * val_x[i];
    }

    //规约求和
    __local float Ssquared[BLOCK_SIZE];
    Ssquared[local_id] = squared;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            Ssquared[local_id] += Ssquared[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //求均方根
    __local float rms;
    if (local_id == 0) {
        rms = native_rsqrt(Ssquared[0] / d + epsilon);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //计算最终结果并存储
    for (int i = 0; i < items; i++) {
        if (local_id * items + i < d)
            y[idx_y + i] = rms * val_x[i] * val_w[i];
    }
}