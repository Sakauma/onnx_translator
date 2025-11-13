#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // 包含 fmaxf

/**
 * CUDA核函数：计算float32类型数据的RELU值
 *
 * @param input 输入数据指针
 * @param output 输出数据指针
 * @param size 数据元素个数
 */
__global__ void relu_kernel_float32(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU的核心逻辑: fmaxf(0.0f, input[idx])
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * 主函数：演示CUDA ReLU计算功能
 */
int main(int argc, char** argv) {
    // 一元操作接收 4 个参数
    if (argc != 4) {
        fprintf(stderr, "用法: %s <元素数量> <输入文件名> <输出文件名>\n", argv[0]);
        return 1;
    }

    size_t num_elements = atol(argv[1]);
    const char* input_filename = argv[2];
    const char* output_filename = argv[3];
    size_t data_size = num_elements * sizeof(float);

    // 1. 从文件读取主机端输入数据
    float* h_input = (float*)malloc(data_size);
    FILE* fp_in = fopen(input_filename, "rb");
    if (!fp_in || fread(h_input, 1, data_size, fp_in) != data_size) {
        fprintf(stderr, "读取输入文件时出错: %s\n", input_filename);
        free(h_input);
        return 1;
    }
    fclose(fp_in);

    // 2. 在GPU上分配内存
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, data_size);
    cudaMalloc((void**)&d_output, data_size);

    // 3. 将主机数据复制到设备
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);

    // 4. 配置并启动CUDA核函数 (调用 relu_kernel_float32)
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    relu_kernel_float32<<<blocks, threads>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();

    // 5. 将结果复制回主机
    float* h_result = (float*)malloc(data_size);
    cudaMemcpy(h_result, d_output, data_size, cudaMemcpyDeviceToHost);

    // 6. 将主机端结果写入文件
    FILE* fp_out = fopen(output_filename, "wb");
    if (!fp_out || fwrite(h_result, 1, data_size, fp_out) != data_size) {
        fprintf(stderr, "写入输出文件时出错: %s\n", output_filename);
        free(h_input);
        free(h_result);
        return 1;
    }
    fclose(fp_out);

    // 7. 释放资源
    free(h_input);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}