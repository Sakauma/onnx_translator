/**
  ******************************************************************************
  * @file        verify_nonzero.cu
  * @author      Egor Izmaylov
  * @brief       提供 nonzero 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 实现 `nonzero_kernel_deterministic` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void nonzero_kernel_deterministic(
    const float* x,
    int64_t* out,
    const int32_t* dims,
    int rank,
    int total,
    int nz_count
) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= total) return;

    if (x[tid] == 0.0f) return;

    // 计算该 nonzero 元素在“按扁平索引升序”的第几个位置
    int pos = 0;
    for (int i = 0; i < tid; ++i) {
        if (x[i] != 0.0f) pos++;
    }

    if (pos >= nz_count) return;

    int rem = tid;
    for (int d = rank - 1; d >= 0; --d) {
        int coord = rem % dims[d];
        rem /= dims[d];
        out[(size_t)d * nz_count + pos] = (int64_t)coord;
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <x.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* p_path = argv[3];
    const char* out_path = argv[4];

    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }

    int32_t rank = 0;
    if (fread(&rank, sizeof(int32_t), 1, fp) != 1) {
        fprintf(stderr, "read rank failed\n");
        verify_close_file(fp);
        return 1;
    }
    if (rank <= 0 || rank > 8) {
        fprintf(stderr, "unsupported rank=%d\n", rank);
        verify_close_file(fp);
        return 1;
    }

    std::vector<int32_t> h_dims(rank);
    if (fread(h_dims.data(), sizeof(int32_t), rank, fp) != (size_t)rank) {
        fprintf(stderr, "read dims failed\n");
        verify_close_file(fp);
        return 1;
    }
    verify_close_file(fp);

    int total = 1;
    for (int i = 0; i < rank; ++i) total *= h_dims[i];

    if (out_len % (size_t)rank != 0) {
        fprintf(stderr, "out_len must be divisible by rank\n");
        return 1;
    }
    int nz_count = (int)(out_len / (size_t)rank);

    std::vector<float> h_x(total);
    FILE* fx = fopen(x_path, "rb");
    if (!fx) {
        fprintf(stderr, "open x failed\n");
        return 1;
    }
    if (fread(h_x.data(), sizeof(float), total, fx) != (size_t)total) {
        fprintf(stderr, "read x failed\n");
        verify_close_file(fx);
        return 1;
    }
    verify_close_file(fx);

    float* d_x = NULL;
    int64_t* d_out = NULL;
    int32_t* d_dims = NULL;

    CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_len * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_dims, rank * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dims, h_dims.data(), rank * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_len * sizeof(int64_t)));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nonzero_kernel_deterministic<<<blocks, threads>>>(d_x, d_out, d_dims, rank, total, nz_count);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int64_t> h_out(out_len);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_len * sizeof(int64_t), cudaMemcpyDeviceToHost));

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open out failed\n");
        return 1;
    }
    if (fwrite(h_out.data(), sizeof(int64_t), out_len, fo) != out_len) {
        fprintf(stderr, "write out failed\n");
        verify_close_file(fo);
        return 1;
    }
    verify_close_file(fo);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_dims));
    return 0;
}