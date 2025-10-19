// verify.cu - CUDA验证程序，用于测试张量操作的正确性
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "tensor_ops.h"
#include <math.h>

/**
 * CUDA核函数：计算float32类型数据的余弦值
 * 
 * @param input 输入数据指针
 * @param output 输出数据指针
 * @param size 数据元素个数
 */
__global__ void cosine_kernel_float32(const float* input, float* output, size_t size) {
    // 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查，确保线程索引在有效范围内
    if (idx < size) {
        // 使用CUDA内置函数计算余弦值
        output[idx] = cosf(input[idx]);
    }
}

/**
 * 主函数：演示CUDA余弦计算功能
 * 
 * @return 程序退出状态码
 */
int main() {
    // 定义数据元素数量
    size_t num_elements = 100;
    
    // 创建张量结构体并初始化
    Tensor t;
    t.size = num_elements;         // 设置元素总数
    t.ndim = 1;                    // 设置为1维张量
    int shape[1] = { (int)num_elements };  // 设置形状数组
    t.shape = shape;               // 指向形状数组
    t.dtype = DTYPE_FLOAT32;       // 设置数据类型为float32
    
    // 分配并初始化主机端数据
    float* h_data = (float*)malloc(num_elements * sizeof(float));
    for (size_t i = 0; i < num_elements; i++) {
        // 初始化数据为 0, 0.1, 0.2, 0.3...
        h_data[i] = (float)i * 0.1f;
    }
    t.data = h_data;  // 将数据指针赋给张量
    
    // 声明设备端输入和输出指针
    float *d_input, *d_output;
    
    // 在GPU上分配内存
    cudaMalloc((void**)&d_input, num_elements * sizeof(float));
    cudaMalloc((void**)&d_output, num_elements * sizeof(float));
    
    // 将主机端数据复制到设备端
    cudaMemcpy(d_input, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    // 配置CUDA执行参数
    int threads = 256;  // 每个线程块的线程数
    int blocks = (num_elements + threads - 1) / threads;  // 计算需要的线程块数
    
    // 启动CUDA核函数
    cosine_kernel_float32<<<blocks, threads>>>(d_input, d_output, num_elements);
    
    // 等待GPU计算完成
    cudaDeviceSynchronize();
    
    // 分配主机端结果内存
    float* h_result = (float*)malloc(num_elements * sizeof(float));
    
    // 将计算结果从设备端复制回主机端
    cudaMemcpy(h_result, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印前10个计算结果
    printf("前10个cos值:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_result[i]);
    }
    printf("\n");
    
    // 释放内存资源
    free(h_data);      // 释放主机端输入数据
    free(h_result);    // 释放主机端输出数据
    cudaFree(d_input); // 释放设备端输入数据
    cudaFree(d_output); // 释放设备端输出数据
    
    return 0;  // 程序正常退出
}
