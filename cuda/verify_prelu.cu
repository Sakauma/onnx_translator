/**
  ******************************************************************************
  * @file        verify_prelu.cu
  * @author      Egor Izmaylov
  * @brief       提供 prelu 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// PRelu 对已按 ONNX 广播规则展开的输入和 slope 逐元素计算 x >= 0 ? x : x * slope。
__global__ void prelu_kernel(const float* input, const float* slope, float* output, size_t n) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < n) {
        float x = input[t];
        output[t] = (x >= 0.0f) ? x : x * slope[t];
    }
}

// 读取一个 float32 输入文件。
static int read_float_input(const char* path, float* dst, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed: %s\n", path);
        return 0;
    }
    size_t r = fread(dst, sizeof(float), n, fp);
    fclose(fp);
    if (r != n) {
        fprintf(stderr, "fread mismatch: %s\n", path);
        return 0;
    }
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <slope.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <slope.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    size_t bytes = n * sizeof(float);
    float* h_input = (float*)malloc(bytes);
    float* h_slope = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_slope || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    if (!read_float_input(argv[2], h_input, n) || !read_float_input(argv[3], h_slope, n)) {
        free(h_input);
        free(h_slope);
        free(h_output);
        return 1;
    }

    float* d_input = NULL;
    float* d_slope = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_slope, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slope, h_slope, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    prelu_kernel<<<blocks, threads>>>(d_input, d_slope, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[4], "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_output, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_slope);
    cudaFree(d_output);
    free(h_input);
    free(h_slope);
    free(h_output);
    return 0;
}
