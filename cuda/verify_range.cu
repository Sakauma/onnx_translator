/**
  ******************************************************************************
  * @file        verify_range.cu
  * @author      Egor Izmaylov
  * @brief       提供 Range 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void range_kernel(float start, float delta, float* output, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid < out_len) {
        output[tid] = start + (float)tid * delta;
    }
}

static int read_scalar(const char* path, float* value) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open scalar failed\n");
        return 0;
    }
    int ok = fread(value, sizeof(float), 1, fp) == 1;
    fclose(fp);
    if (!ok) {
        fprintf(stderr, "read scalar failed\n");
    }
    return ok;
}

int main(int argc, char** argv) {
    // <out_len> <start.bin> <limit.bin> <delta.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <start.bin> <limit.bin> <delta.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* start_path = argv[2];
    const char* delta_path = argv[4];
    const char* out_path = argv[5];

    float start = 0.0f;
    float delta = 0.0f;
    if (!read_scalar(start_path, &start) || !read_scalar(delta_path, &delta)) {
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
    range_kernel<<<blocks, threads>>>(start, delta, d_output, out_len);
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
