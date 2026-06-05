/**
  ******************************************************************************
  * @file        verify_bitwise_not.cu
  * @author      Egor Izmaylov
  * @brief       提供 BitwiseNot 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 按 int32 位模式执行逐元素按位取反。
__global__ void bitwise_not_kernel(const int32_t* input, int32_t* output, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;
    output[tid] = ~input[tid];
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <out.bin>
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* out_path = argv[3];
    size_t bytes = out_len * sizeof(int32_t);

    int32_t* h_input = (int32_t*)malloc(bytes);
    int32_t* h_output = (int32_t*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* input_fp = fopen(input_path, "rb");
    if (!input_fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t read_count = fread(h_input, sizeof(int32_t), out_len, input_fp);
    fclose(input_fp);
    if (read_count != out_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    int32_t* d_input = NULL;
    int32_t* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    bitwise_not_kernel<<<blocks, threads>>>(d_input, d_output, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t written = fwrite(h_output, sizeof(int32_t), out_len, out_fp);
    fclose(out_fp);
    if (written != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
