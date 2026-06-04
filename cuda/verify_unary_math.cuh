/**
  ******************************************************************************
  * @file        verify_unary_math.cuh
  * @author      Egor Izmaylov
  * @brief       提供一元数学 CUDA verifier 的共享模板。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef VERIFY_UNARY_MATH_CUH
#define VERIFY_UNARY_MATH_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define VERIFY_CONCAT2(a, b) a##b
#define VERIFY_CONCAT(a, b) VERIFY_CONCAT2(a, b)
#define VERIFY_STRINGIFY2(x) #x
#define VERIFY_STRINGIFY(x) VERIFY_STRINGIFY2(x)

#ifndef VERIFY_OP_NAME
#error "VERIFY_OP_NAME must be defined before including verify_unary_math.cuh"
#endif

#ifndef VERIFY_EXPR
#error "VERIFY_EXPR(x) must be defined before including verify_unary_math.cuh"
#endif

// 将一元数学函数逐元素应用到输入张量。
__global__ void VERIFY_CONCAT(VERIFY_OP_NAME, _kernel)(const float* in, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = VERIFY_EXPR(x);
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
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "malloc failed for %s\n", VERIFY_STRINGIFY(VERIFY_OP_NAME));
        return 1;
    }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t r = fread(h_in, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    VERIFY_CONCAT(VERIFY_OP_NAME, _kernel)<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}

#endif
