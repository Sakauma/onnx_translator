// verify_cos.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "tensor_ops/tensor_ops.h"
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
int main(int argc, char** argv) {
    // // 定义数据元素数量
    // size_t num_elements = 100;
    
    // // 创建张量结构体并初始化
    // Tensor t;
    // t.size = num_elements;         // 设置元素总数
    // t.ndim = 1;                    // 设置为1维张量
    // int shape[1] = { (int)num_elements };  // 设置形状数组
    // t.shape = shape;               // 指向形状数组
    // t.dtype = DTYPE_FLOAT32;       // 设置数据类型为float32
    
    // // 分配并初始化主机端数据
    // float* h_data = (float*)malloc(num_elements * sizeof(float));
    // for (size_t i = 0; i < num_elements; i++) {
    //     // 初始化数据为 0, 0.1, 0.2, 0.3...
    //     h_data[i] = (float)i * 0.1f;
    // }
    // t.data = h_data;  // 将数据指针赋给张量
    
    // // 声明设备端输入和输出指针
    // float *d_input, *d_output;
    
    // // 在GPU上分配内存
    // cudaMalloc((void**)&d_input, num_elements * sizeof(float));
    // cudaMalloc((void**)&d_output, num_elements * sizeof(float));
    
    // // 将主机端数据复制到设备端
    // cudaMemcpy(d_input, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    // // 配置CUDA执行参数
    // int threads = 256;  // 每个线程块的线程数
    // int blocks = (num_elements + threads - 1) / threads;  // 计算需要的线程块数
    
    // // 启动CUDA核函数
    // cosine_kernel_float32<<<blocks, threads>>>(d_input, d_output, num_elements);
    
    // // 等待GPU计算完成
    // cudaDeviceSynchronize();
    
    // // 分配主机端结果内存
    // float* h_result = (float*)malloc(num_elements * sizeof(float));
    
    // // 将计算结果从设备端复制回主机端
    // cudaMemcpy(h_result, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // // 打印前10个计算结果
    // printf("前10个cos值:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", h_result[i]);
    // }
    // printf("\n");
    
    // // 释放内存资源
    // free(h_data);      // 释放主机端输入数据
    // free(h_result);    // 释放主机端输出数据
    // cudaFree(d_input); // 释放设备端输入数据
    // cudaFree(d_output); // 释放设备端输出数据
    
    // return 0;  // 程序正常退出

    // 检查命令行参数是否正确
    if (argc != 4) {
        fprintf(stderr, "用法: %s <元素数量> <输入文件名> <输出文件名>\n", argv[0]);
        return 1;
    }

    // 解析命令行参数
    size_t num_elements = atol(argv[1]);
    const char* input_filename = argv[2];
    const char* output_filename = argv[3];
    size_t data_size = num_elements * sizeof(float);

    // 1. 分配主机内存并从输入文件读取数据
    float* h_input = (float*)malloc(data_size);
    FILE* fp_in = fopen(input_filename, "rb");
    if (!fp_in || fread(h_input, 1, data_size, fp_in) != data_size) {
        fprintf(stderr, "读取输入文件时出错: %s\n", input_filename);
        free(h_input);
        return 1;
    }
    fclose(fp_in);

    // 2. 分配 GPU 内存
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, data_size);
    cudaMalloc((void**)&d_output, data_size);

    // 3. 将数据从主机复制到设备
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);

    // 4. 配置并启动 CUDA 核函数
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    cosine_kernel_float32<<<blocks, threads>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();

    // 5. 将结果从设备复制回主机
    float* h_result = (float*)malloc(data_size);
    cudaMemcpy(h_result, d_output, data_size, cudaMemcpyDeviceToHost);

    // 6. 将主机端的结果写入输出文件
    FILE* fp_out = fopen(output_filename, "wb");
    if (!fp_out || fwrite(h_result, 1, data_size, fp_out) != data_size) {
        fprintf(stderr, "写入输出文件时出错: %s\n", output_filename);
        free(h_input);
        free(h_result);
        return 1;
    }
    fclose(fp_out);

    // 7. 释放所有已分配的内存
    free(h_input);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
