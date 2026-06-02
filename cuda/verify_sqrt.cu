/**
  ******************************************************************************
  * @file        verify_sqrt.cu
  * @author      Egor Izmaylov
  * @brief       提供 sqrt 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 实现 `sqrt_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void sqrt_kernel(const float* a, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 主路径：输入非负
        out[idx] = sqrtf(a[idx]);
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <in0.bin> <out.bin>
    if (argc != 4) {
        printf("Usage: %s <out_len> <in0.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* in_path  = argv[2];
    const char* out_path = argv[3];

    size_t bytes = n * sizeof(float);

    float* h_a   = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fi = fopen(in_path, "rb");
    if (!fi) { printf("open input failed\n"); return 1; }
    size_t r = fread(h_a, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) { printf("fread mismatch\n"); return 1; }

    float *d_a = NULL, *d_out = NULL;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    sqrt_kernel<<<blocks, threads>>>(d_a, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_a);
    cudaFree(d_out);
    free(h_a);
    free(h_out);
    return 0;
}
