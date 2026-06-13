/**
  ******************************************************************************
  * @file        verify_blackman_window.cu
  * @author      Egor Izmaylov
  * @brief       提供 BlackmanWindow 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void blackman_window_kernel(float* output, int32_t size, int32_t periodic) {
    int32_t i = (int32_t)blockIdx.x * (int32_t)blockDim.x + (int32_t)threadIdx.x;
    if (i >= size) return;
    if (size <= 0) return;
    if (size == 1) {
        output[i] = 1.0f;
        return;
    }

    double denom = (double)(periodic ? size : size - 1);
    double angle = 2.0 * M_PI * (double)i / denom;
    output[i] = (float)(0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle));
}

int main(int argc, char** argv) {
    // <out_len> <size.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <size.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    int32_t params[2] = {0, 1};
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(params, sizeof(int32_t), 2, fp) != 2) {
        fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 1;
    }
    fclose(fp);

    int32_t size = params[0];
    int32_t periodic = params[1];
    if (size < 0 || out_len != (size_t)size) {
        fprintf(stderr, "size mismatch\n");
        return 1;
    }

    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    float* d_output = NULL;
    cudaMalloc(&d_output, out_len * sizeof(float));
    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    blackman_window_kernel<<<blocks, threads>>>(d_output, size, periodic);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t write_count = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_output);
    free(h_output);
    return 0;
}
