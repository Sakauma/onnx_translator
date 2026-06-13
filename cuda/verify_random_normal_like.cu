/**
  ******************************************************************************
  * @file        verify_random_normal_like.cu
  * @author      Egor Izmaylov
  * @brief       提供 RandomNormalLike 算子的 CUDA 参考验证程序。
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

struct RandomNormalLikeParams {
    int32_t numel;
    float mean;
    float scale;
    uint32_t seed;
};

// 按元素生成正态分布样本，输入张量只用于提供 Like 算子的输出形状。
__global__ void random_normal_like_kernel(float* out, RandomNormalLikeParams params) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= params.numel) return;
    double z0 = verify_random_normal01(params.seed, tid);
    out[tid] = (float)((double)params.mean + z0 * (double)params.scale);
}

// 从 params 二进制文件读取 RandomNormalLike 参数。
static int read_params(const char* path, RandomNormalLikeParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int ok = fread(params, sizeof(RandomNormalLikeParams), 1, fp) == 1;
    fclose(fp);
    return ok;
}

// 将 float32 输出写入二进制文件。
static int write_f32_file(const char* path, const float* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(float), n, fp);
    fclose(fp);
    return wrote == n;
}

// 作为 CUDA 验证程序入口，执行 RandomNormalLike 参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <dummy_input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    RandomNormalLikeParams params;
    if (!read_params(argv[3], &params)) return 1;
    if (params.numel < 0 || (size_t)params.numel != out_len) return 1;

    float* h_out = (float*)malloc(out_len * sizeof(float));
    float* d_out = NULL;
    if (!h_out) return 1;
    cudaMalloc(&d_out, out_len * sizeof(float));

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    random_normal_like_kernel<<<blocks, threads>>>(d_out, params);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    int ok = write_f32_file(argv[4], h_out, out_len);
    cudaFree(d_out);
    free(h_out);
    return ok ? 0 : 1;
}
