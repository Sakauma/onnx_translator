/*
 * 文件功能：提供 xor 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// 实现 `xor_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void xor_kernel(const float* A, const float* B, unsigned char* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float a = A[idx];
        float b = B[idx];
        out[idx] = (unsigned char)(((a!=0.0f)^(b!=0.0f)) ? 1 : 0);
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <n> <a.bin> <b.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <n> <a.bin> <b.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoll(argv[1]);
    size_t in_bytes = n * sizeof(float);
    size_t out_bytes = n * sizeof(unsigned char);

    float* h_a  = (float*)malloc(in_bytes);
    float* h_b  = (float*)malloc(in_bytes);
    unsigned char* h_out= (unsigned char*)malloc(out_bytes);
    if (!h_a || !h_b || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fa = fopen(argv[2], "rb");
    FILE* fb = fopen(argv[3], "rb");
    if (!fa || !fb) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t ra = fread(h_a, sizeof(float), n, fa);
    size_t rb = fread(h_b, sizeof(float), n, fb);
    fclose(fa); fclose(fb);
    if (ra != n || rb != n) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_a=NULL, *d_b=NULL; unsigned char* d_out=NULL;
    cudaMalloc(&d_a, in_bytes);
    cudaMalloc(&d_b, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_a, h_a, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    xor_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[4], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(unsigned char), n, fo);
    fclose(fo);
    if (w != n) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}
