/**
  ******************************************************************************
  * @file        verify_reduce_log_sum_exp.cu
  * @author      Egor Izmaylov
  * @brief       提供 reduce_log_sum_exp 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <cuda_runtime.h>
#include "verify_common.cuh"

typedef struct { int64_t in_len; } ReduceAllParams;

// 使用稳定公式 log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))。
__global__ void reduce_log_sum_exp_kernel(const float* in, float* out, int64_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double max_val = -DBL_MAX;
    bool has_nan = false;
    bool has_positive_infinity = false;
    for (int64_t i = 0; i < n; ++i) {
        double v = (double)in[i];
        if (isnan(v)) has_nan = true;
        if (isinf(v) && v > 0.0) has_positive_infinity = true;
        if (v > max_val) max_val = v;
    }
    if (has_nan) {
        out[0] = NAN;
        return;
    }
    if (has_positive_infinity) {
        out[0] = INFINITY;
        return;
    }
    if (isinf(max_val) && max_val < 0.0) {
        out[0] = -INFINITY;
        return;
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += exp((double)in[i] - max_val);
    }
    out[0] = (float)(log(sum) + max_val);
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <in.bin> <params.bin> <out.bin>
    if (argc != 5) return 1;
    size_t out_len = (size_t)atoll(argv[1]);
    if (out_len != 1) return 1;

    ReduceAllParams p;
    verify_read_file(argv[3], &p, sizeof(p));
    if (p.in_len <= 0) return 1;

    size_t in_len = (size_t)p.in_len;
    size_t in_bytes = in_len * sizeof(float);
    float* h_in = (float*)verify_malloc(in_bytes);
    float h_out = 0.0f;
    verify_read_file(argv[2], h_in, in_bytes);

    float *d_in = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    reduce_log_sum_exp_kernel<<<1, 1>>>(d_in, d_out, p.in_len);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    verify_write_file(argv[4], &h_out, sizeof(h_out));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    return 0;
}
