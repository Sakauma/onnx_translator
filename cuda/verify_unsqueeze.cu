/**
  ******************************************************************************
  * @file        verify_unsqueeze.cu
  * @author      Egor Izmaylov
  * @brief       提供 Unsqueeze 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Unsqueeze 只插入长度为 1 的维度，不改变连续内存中的元素顺序。
__global__ void unsqueeze_kernel(const float* input, float* output, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <out.bin>
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* out_path = argv[3];
    size_t bytes = n * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), n, fi);
    fclose(fi);
    if (ri != n) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    unsqueeze_kernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(float), n, fo);
    fclose(fo);
    if (wo != n) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
