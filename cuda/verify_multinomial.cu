/**
  ******************************************************************************
  * @file        verify_multinomial.cu
  * @author      Egor Izmaylov
  * @brief       提供 Multinomial 算子的 CUDA 参考验证程序。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "verify_random_common.cuh"

struct MultinomialParams {
    int32_t batch;
    int32_t classes;
    int32_t sample_size;
    int32_t output_dtype_code;
    uint32_t seed;
};

// 对每一行概率分布独立采样，复现 C 后端按行 seed 和 simple_lcg 的选择逻辑。
__global__ void multinomial_kernel(const float* probs, int64_t* out_i64, int32_t* out_i32, MultinomialParams params) {
    int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= params.batch) return;

    double total = 0.0;
    for (int c = 0; c < params.classes; ++c) {
        double p = (double)probs[row * params.classes + c];
        if (p > 0.0) total += p;
    }
    if (total <= 0.0) return;

    uint32_t state = params.seed ? (params.seed + (uint32_t)row * 747796405u) : (uint32_t)row;
    for (int sample = 0; sample < params.sample_size; ++sample) {
        state = verify_random_lcg_next(state);
        double threshold = ((double)state / 2147483648.0) * total;
        double cumulative = 0.0;
        int selected = params.classes - 1;
        for (int c = 0; c < params.classes; ++c) {
            double p = (double)probs[row * params.classes + c];
            if (p <= 0.0) continue;
            cumulative += p;
            if (threshold < cumulative) {
                selected = c;
                break;
            }
        }
        int out_idx = row * params.sample_size + sample;
        if (params.output_dtype_code == 1) {
            out_i64[out_idx] = (int64_t)selected;
        } else {
            out_i32[out_idx] = (int32_t)selected;
        }
    }
}

// 从 params 二进制文件读取 Multinomial 参数。
static int read_params(const char* path, MultinomialParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int ok = fread(params, sizeof(MultinomialParams), 1, fp) == 1;
    fclose(fp);
    return ok;
}

// 读取 float32 输入概率文件。
static int read_f32_file(const char* path, float* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t got = fread(data, sizeof(float), n, fp);
    fclose(fp);
    return got == n;
}

// 写回 int64 输出文件。
static int write_i64_file(const char* path, const int64_t* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(int64_t), n, fp);
    fclose(fp);
    return wrote == n;
}

// 写回 int32 输出文件。
static int write_i32_file(const char* path, const int32_t* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(int32_t), n, fp);
    fclose(fp);
    return wrote == n;
}

// 作为 CUDA 验证程序入口，执行 Multinomial 参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    MultinomialParams params;
    if (!read_params(argv[3], &params)) return 1;
    if (params.batch < 0 || params.classes <= 0 || params.sample_size < 0) return 1;
    if ((size_t)params.batch * (size_t)params.sample_size != out_len) return 1;

    size_t input_len = (size_t)params.batch * (size_t)params.classes;
    float* h_probs = (float*)malloc(input_len * sizeof(float));
    int64_t* h_out_i64 = NULL;
    int32_t* h_out_i32 = NULL;
    float* d_probs = NULL;
    int64_t* d_out_i64 = NULL;
    int32_t* d_out_i32 = NULL;
    if (!h_probs) return 1;
    if (!read_f32_file(argv[2], h_probs, input_len)) return 1;

    cudaMalloc(&d_probs, input_len * sizeof(float));
    cudaMemcpy(d_probs, h_probs, input_len * sizeof(float), cudaMemcpyHostToDevice);

    if (params.output_dtype_code == 1) {
        h_out_i64 = (int64_t*)calloc(out_len, sizeof(int64_t));
        if (!h_out_i64) return 1;
        cudaMalloc(&d_out_i64, out_len * sizeof(int64_t));
        cudaMemset(d_out_i64, 0, out_len * sizeof(int64_t));
    } else {
        h_out_i32 = (int32_t*)calloc(out_len, sizeof(int32_t));
        if (!h_out_i32) return 1;
        cudaMalloc(&d_out_i32, out_len * sizeof(int32_t));
        cudaMemset(d_out_i32, 0, out_len * sizeof(int32_t));
    }

    int threads = 128;
    int blocks = (params.batch + threads - 1) / threads;
    multinomial_kernel<<<blocks, threads>>>(d_probs, d_out_i64, d_out_i32, params);
    cudaDeviceSynchronize();

    int ok = 0;
    if (params.output_dtype_code == 1) {
        cudaMemcpy(h_out_i64, d_out_i64, out_len * sizeof(int64_t), cudaMemcpyDeviceToHost);
        ok = write_i64_file(argv[4], h_out_i64, out_len);
    } else {
        cudaMemcpy(h_out_i32, d_out_i32, out_len * sizeof(int32_t), cudaMemcpyDeviceToHost);
        ok = write_i32_file(argv[4], h_out_i32, out_len);
    }

    cudaFree(d_probs);
    if (d_out_i64) cudaFree(d_out_i64);
    if (d_out_i32) cudaFree(d_out_i32);
    free(h_probs);
    free(h_out_i64);
    free(h_out_i32);
    return ok ? 0 : 1;
}
