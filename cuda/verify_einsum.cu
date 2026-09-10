/**
  ******************************************************************************
  * @file        verify_einsum.cu
  * @author      Egor Izmaylov
  * @brief       提供 einsum 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>
#include "verify_common.cuh"

struct EinsumParams {
    int32_t M;
    int32_t K;
    int32_t N;
};

// 实现 `einsum_ij_jk_to_ik_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void einsum_ij_jk_to_ik_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = M * N;
    if (tid >= total) return;

    int i = tid / N;
    int j = tid % N;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[(size_t)i * K + k] * B[(size_t)k * N + j];
    }
    C[tid] = sum;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <A.bin> <B.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <A.bin> <B.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* a_path = argv[2];
    const char* b_path = argv[3];
    const char* p_path = argv[4];
    const char* out_path = argv[5];

    EinsumParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(EinsumParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        verify_close_file(fp);
        return 1;
    }
    verify_close_file(fp);

    if ((size_t)p.M * p.N != out_len) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    size_t a_len = (size_t)p.M * p.K;
    size_t b_len = (size_t)p.K * p.N;

    std::vector<float> h_A(a_len);
    std::vector<float> h_B(b_len);

    FILE* fa = fopen(a_path, "rb");
    FILE* fb = fopen(b_path, "rb");
    if (!fa || !fb) {
        fprintf(stderr, "open inputs failed\n");
        return 1;
    }

    if (fread(h_A.data(), sizeof(float), a_len, fa) != a_len) {
        fprintf(stderr, "read A failed\n");
        verify_close_file(fa);
        verify_close_file(fb);
        return 1;
    }
    if (fread(h_B.data(), sizeof(float), b_len, fb) != b_len) {
        fprintf(stderr, "read B failed\n");
        verify_close_file(fa);
        verify_close_file(fb);
        return 1;
    }
    verify_close_file(fa);
    verify_close_file(fb);

    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    CUDA_CHECK(cudaMalloc(&d_A, a_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, out_len * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_len * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    einsum_ij_jk_to_ik_kernel<<<blocks, threads>>>(d_A, d_B, d_C, p.M, p.K, p.N);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(out_len);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, out_len * sizeof(float), cudaMemcpyDeviceToHost));

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open out failed\n");
        return 1;
    }
    if (fwrite(h_C.data(), sizeof(float), out_len, fo) != out_len) {
        fprintf(stderr, "write out failed\n");
        verify_close_file(fo);
        return 1;
    }
    verify_close_file(fo);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}