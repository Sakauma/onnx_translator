/**
  ******************************************************************************
  * @file        verify_where.cu
  * @author      Egor Izmaylov
  * @brief       提供 Where 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 按条件逐元素选择 X 或 Y，输入已经由 runner 按 ONNX 广播规则物化。
__global__ void where_kernel(const float* cond, const float* x, const float* y, float* output, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        output[idx] = cond[idx] != 0.0f ? x[idx] : y[idx];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <cond.bin> <x.bin> <y.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <cond.bin> <x.bin> <y.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* cond_path = argv[2];
    const char* x_path = argv[3];
    const char* y_path = argv[4];
    const char* out_path = argv[5];
    size_t bytes = n * sizeof(float);

    float* h_cond = (float*)malloc(bytes);
    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_cond || !h_x || !h_y || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fc = fopen(cond_path, "rb");
    FILE* fx = fopen(x_path, "rb");
    FILE* fy = fopen(y_path, "rb");
    if (!fc || !fx || !fy) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t rc = fread(h_cond, sizeof(float), n, fc);
    size_t rx = fread(h_x, sizeof(float), n, fx);
    size_t ry = fread(h_y, sizeof(float), n, fy);
    fclose(fc);
    fclose(fx);
    fclose(fy);
    if (rc != n || rx != n || ry != n) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_cond = NULL;
    float* d_x = NULL;
    float* d_y = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_cond, bytes);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_cond, h_cond, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    where_kernel<<<blocks, threads>>>(d_cond, d_x, d_y, d_output, n);
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

    cudaFree(d_cond);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_output);
    free(h_cond);
    free(h_x);
    free(h_y);
    free(h_output);
    return 0;
}
