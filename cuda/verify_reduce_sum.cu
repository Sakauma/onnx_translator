/**
  ******************************************************************************
  * @file        verify_reduce_sum.cu
  * @author      Egor Izmaylov
  * @brief       提供 reduce sum 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

typedef struct { int64_t in_len; } ReduceAllParams;

// 实现 `reduce_sum_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void reduce_sum_kernel(const float* in, float* out, int64_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc = acc + (double)in[i];
    }
    out[0] = (float)acc;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <in.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <in.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    if (out_len != 1) {
        fprintf(stderr, "This verifier expects out_len==1 (reduce-all). Got %zu\n", out_len);
        return 1;
    }

    ReduceAllParams p;
    FILE* fp = fopen(argv[3], "rb");
    if (!fp) { fprintf(stderr, "open params failed\n"); return 1; }
    size_t pr = fread(&p, sizeof(ReduceAllParams), 1, fp);
    fclose(fp);
    if (pr != 1 || p.in_len <= 0) {
        fprintf(stderr, "read params failed\n");
        return 1;
    }

    size_t in_len = (size_t)p.in_len;
    size_t in_bytes = in_len * sizeof(float);
    size_t out_bytes = sizeof(float);

    float* h_in = (float*)malloc(in_bytes);
    float h_out = 0.0f;
    if (!h_in) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t r = fread(h_in, sizeof(float), in_len, fi);
    fclose(fi);
    if (r != in_len) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    reduce_sum_kernel<<<1, 1>>>(d_in, d_out, p.in_len);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[4], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(&h_out, sizeof(float), 1, fo);
    fclose(fo);
    if (w != 1) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
