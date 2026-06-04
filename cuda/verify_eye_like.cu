/**
  ******************************************************************************
  * @file        verify_eye_like.cu
  * @author      Egor Izmaylov
  * @brief       提供 eye_like 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// EyeLike 根据输入形状生成二维单位矩阵，可通过 k 偏移对角线。
__global__ void eye_like_kernel(float* output, int rows, int cols, int k, size_t out_len) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= out_len) return;

    int row = (int)(t / (size_t)cols);
    int col = (int)(t % (size_t)cols);
    output[t] = (col == row + k) ? 1.0f : 0.0f;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        printf("open params failed\n");
        return 1;
    }

    int params[3] = {0};
    if (fread(params, sizeof(int), 3, fp) != 3) {
        fclose(fp);
        printf("read params failed\n");
        return 1;
    }
    fclose(fp);

    int rows = params[0];
    int cols = params[1];
    int k = params[2];
    if (rows <= 0 || cols <= 0) {
        printf("invalid shape\n");
        return 1;
    }

    size_t expected_out_len = (size_t)rows * (size_t)cols;
    if (out_len != expected_out_len) {
        printf("out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    size_t out_bytes = out_len * sizeof(float);
    float* h_output = (float*)malloc(out_bytes);
    if (!h_output) {
        printf("malloc failed\n");
        return 1;
    }

    float* d_output = NULL;
    cudaMalloc(&d_output, out_bytes);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    eye_like_kernel<<<blocks, threads>>>(d_output, rows, cols, k, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        printf("open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) {
        printf("fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_output);
    free(h_output);
    return 0;
}
