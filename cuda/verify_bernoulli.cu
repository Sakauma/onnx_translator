/**
  ******************************************************************************
  * @file        verify_bernoulli.cu
  * @author      Egor Izmaylov
  * @brief       提供 Bernoulli 算子的 CUDA 参考验证程序。
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

struct BernoulliParams {
    int32_t numel;
    uint32_t seed;
};

// 根据概率张量逐元素采样 0/1，参考 C 后端确定性随机公式。
__global__ void bernoulli_kernel(const float* probs, float* out, BernoulliParams params) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= params.numel) return;
    double u = verify_random_uniform01(params.seed, tid);
    out[tid] = (u < (double)probs[tid]) ? 1.0f : 0.0f;
}

// 从 params 二进制文件读取 Bernoulli 参数。
static int read_params(const char* path, BernoulliParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int ok = fread(params, sizeof(BernoulliParams), 1, fp) == 1;
    fclose(fp);
    return ok;
}

// 读取 float32 输入文件。
static int read_f32_file(const char* path, float* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t got = fread(data, sizeof(float), n, fp);
    fclose(fp);
    return got == n;
}

// 将 float32 输出写入二进制文件。
static int write_f32_file(const char* path, const float* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(float), n, fp);
    fclose(fp);
    return wrote == n;
}

// 作为 CUDA 验证程序入口，执行 Bernoulli 参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    BernoulliParams params;
    if (!read_params(argv[3], &params)) return 1;
    if (params.numel < 0 || (size_t)params.numel != out_len) return 1;

    float* h_probs = (float*)malloc(out_len * sizeof(float));
    float* h_out = (float*)malloc(out_len * sizeof(float));
    float* d_probs = NULL;
    float* d_out = NULL;
    if (!h_probs || !h_out) return 1;
    if (!read_f32_file(argv[2], h_probs, out_len)) return 1;

    cudaMalloc(&d_probs, out_len * sizeof(float));
    cudaMalloc(&d_out, out_len * sizeof(float));
    cudaMemcpy(d_probs, h_probs, out_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    bernoulli_kernel<<<blocks, threads>>>(d_probs, d_out, params);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    int ok = write_f32_file(argv[4], h_out, out_len);
    cudaFree(d_probs);
    cudaFree(d_out);
    free(h_probs);
    free(h_out);
    return ok ? 0 : 1;
}
