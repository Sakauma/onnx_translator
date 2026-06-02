/*
 * 文件功能：提供 atan 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 实现 `atan_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void atan_kernel(const float* in, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = atanf(x);
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <n> <in.bin> <out.bin>
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <n> <in.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoll(argv[1]);
    size_t bytes = n * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t r = fread(h_in, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    atan_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
