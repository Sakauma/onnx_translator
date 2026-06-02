/**
  ******************************************************************************
  * @file        verify_mod.cu
  * @author      Egor Izmaylov
  * @brief       提供 mod 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 实现 `python_mod` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ __forceinline__ float python_mod(float a, float b) {
    if (b == 0.0f) return NAN;
    float r = fmodf(a, b);
    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
    return r;
}

// 实现 `mod_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void mod_kernel(const float* a, const float* b, float* out, int n) {
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (t < n) out[t] = python_mod(a[t], b[t]);
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <a.bin> <b.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <a.bin> <b.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    int out_len = (int)atoll(argv[1]);
    const char* a_path = argv[2];
    const char* b_path = argv[3];
    const char* out_path = argv[4];

    size_t bytes = (size_t)out_len * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fa = fopen(a_path, "rb");
    FILE* fb = fopen(b_path, "rb");
    if (!fa || !fb) { printf("open input failed\n"); return 1; }

    size_t ra = fread(h_a, sizeof(float), out_len, fa);
    size_t rb = fread(h_b, sizeof(float), out_len, fb);
    fclose(fa); fclose(fb);
    if (ra != (size_t)out_len || rb != (size_t)out_len) { printf("fread mismatch\n"); return 1; }

    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    mod_kernel<<<blocks, threads>>>(d_a, d_b, d_out, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t wo = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != (size_t)out_len) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}