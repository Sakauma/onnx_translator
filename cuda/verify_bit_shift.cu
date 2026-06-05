/**
  ******************************************************************************
  * @file        verify_bit_shift.cu
  * @author      Egor Izmaylov
  * @brief       提供 BitShift 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// direction=0 表示左移，direction=1 表示算术右移。
__global__ void bit_shift_kernel(
    const int32_t* lhs,
    const int32_t* rhs,
    int32_t* output,
    int32_t direction,
    size_t out_len
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    int32_t shift = rhs[tid];
    if (shift < 0 || shift >= 64) {
        output[tid] = 0;
        return;
    }
    if (direction == 0) {
        uint64_t raw = (uint64_t)(int64_t)lhs[tid];
        output[tid] = (int32_t)(raw << (uint32_t)shift);
    } else {
        int64_t value = (int64_t)lhs[tid];
        output[tid] = (int32_t)(value >> shift);
    }
}

// 读取 direction 参数。
static int read_direction(const char* params_path, int32_t* direction) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    if (fread(direction, sizeof(int32_t), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    fclose(fp);
    return *direction == 0 || *direction == 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <lhs.bin> <rhs.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <lhs.bin> <rhs.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* lhs_path = argv[2];
    const char* rhs_path = argv[3];
    const char* params_path = argv[4];
    const char* out_path = argv[5];
    size_t bytes = out_len * sizeof(int32_t);

    int32_t direction = 0;
    if (!read_direction(params_path, &direction)) {
        return 1;
    }

    int32_t* h_lhs = (int32_t*)malloc(bytes);
    int32_t* h_rhs = (int32_t*)malloc(bytes);
    int32_t* h_output = (int32_t*)malloc(bytes);
    if (!h_lhs || !h_rhs || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* lhs_fp = fopen(lhs_path, "rb");
    FILE* rhs_fp = fopen(rhs_path, "rb");
    if (!lhs_fp || !rhs_fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t lhs_read = fread(h_lhs, sizeof(int32_t), out_len, lhs_fp);
    size_t rhs_read = fread(h_rhs, sizeof(int32_t), out_len, rhs_fp);
    fclose(lhs_fp);
    fclose(rhs_fp);
    if (lhs_read != out_len || rhs_read != out_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    int32_t* d_lhs = NULL;
    int32_t* d_rhs = NULL;
    int32_t* d_output = NULL;
    cudaMalloc(&d_lhs, bytes);
    cudaMalloc(&d_rhs, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_lhs, h_lhs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, h_rhs, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    bit_shift_kernel<<<blocks, threads>>>(d_lhs, d_rhs, d_output, direction, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t written = fwrite(h_output, sizeof(int32_t), out_len, out_fp);
    fclose(out_fp);
    if (written != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_lhs);
    cudaFree(d_rhs);
    cudaFree(d_output);
    free(h_lhs);
    free(h_rhs);
    free(h_output);
    return 0;
}
