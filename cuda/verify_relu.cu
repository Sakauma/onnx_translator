// verify_add.cu - CUDA验证程序, 用于测试 ADD 操作的正确性
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // 包含 fmaxf

/**
 * CUDA核函数：计算float32类型数据的 ADD (A + B)
 * 假设 A, B, O 具有完全相同的形状和大小
 */
__global__ void add_kernel_float32(const float* input_a, const float* input_b, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ADD 的核心逻辑: A[i] + B[i]
        output[idx] = input_a[idx] + input_b[idx];
    }
}

/**
 * 主函数：从文件读取 A 和 B，计算 A+B，写入文件
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "用法: %s <元素数量> <输入文件A> <输入文件B> <输出文件名>\n", argv[0]);
        return 1;
    }

    size_t num_elements = atol(argv[1]);
    const char* input_filename_a = argv[2];
    const char* input_filename_b = argv[3];
    const char* output_filename = argv[4];
    size_t data_size = num_elements * sizeof(float);

    // 1. 分配主机内存
    float* h_input_a = (float*)malloc(data_size);
    float* h_input_b = (float*)malloc(data_size);
    float* h_result = (float*)malloc(data_size);

    // 2. 读取输入文件 A
    FILE* fp_in_a = fopen(input_filename_a, "rb");
    if (!fp_in_a || fread(h_input_a, 1, data_size, fp_in_a) != data_size) {
        fprintf(stderr, "读取输入文件A时出错: %s\n", input_filename_a);
        free(h_input_a); free(h_input_b); free(h_result);
        return 1;
    }
    fclose(fp_in_a);

    // 3. 读取输入文件 B
    FILE* fp_in_b = fopen(input_filename_b, "rb");
    if (!fp_in_b || fread(h_input_b, 1, data_size, fp_in_b) != data_size) {
        fprintf(stderr, "读取输入文件B时出错: %s\n", input_filename_b);
        free(h_input_a); free(h_input_b); free(h_result);
        return 1;
    }
    fclose(fp_in_b);

    // 4. 在GPU上分配内存 (2个输入, 1个输出)
    float *d_input_a, *d_input_b, *d_output;
    cudaMalloc((void**)&d_input_a, data_size);
    cudaMalloc((void**)&d_input_b, data_size);
    cudaMalloc((void**)&d_output, data_size);

    // 5. 将主机数据复制到设备
    cudaMemcpy(d_input_a, h_input_a, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, h_input_b, data_size, cudaMemcpyHostToDevice);

    // 6. 配置并启动CUDA核函数
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    add_kernel_float32<<<blocks, threads>>>(d_input_a, d_input_b, d_output, num_elements);
    cudaDeviceSynchronize();

    // 7. 将结果复制回主机
    cudaMemcpy(h_result, d_output, data_size, cudaMemcpyDeviceToHost);

    // 8. 将主机端结果写入文件
    FILE* fp_out = fopen(output_filename, "wb");
    if (!fp_out || fwrite(h_result, 1, data_size, fp_out) != data_size) {
        fprintf(stderr, "写入输出文件时出错: %s\n", output_filename);
        return 1;
    }
    fclose(fp_out);

    // 9. 释放资源
    free(h_input_a);
    free(h_input_b);
    free(h_result);
    cudaFree(d_input_a);
    cudaFree(d_input_b);
    cudaFree(d_output);

    return 0;
}