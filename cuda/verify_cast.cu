/**
  ******************************************************************************
  * @file        verify_cast.cu
  * @author      Egor Izmaylov
  * @brief       提供 cast 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CAST_OUT_FLOAT32 0
#define CAST_OUT_INT32 1
#define CAST_OUT_INT64 2
#define CAST_OUT_BOOL 3

// 将 float32 输入按 Cast 主路径转换到目标输出缓冲区。
__global__ void cast_kernel(const float* input, void* output, size_t n, int output_kind) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= n) return;

    float value = input[t];
    if (output_kind == CAST_OUT_INT32) {
        ((int32_t*)output)[t] = isfinite(value) ? (int32_t)value : 0;
    } else if (output_kind == CAST_OUT_INT64) {
        ((int64_t*)output)[t] = isfinite(value) ? (int64_t)value : 0;
    } else if (output_kind == CAST_OUT_BOOL) {
        ((uint8_t*)output)[t] = value != 0.0f;
    } else {
        ((float*)output)[t] = value;
    }
}

// 根据输出类型选择 CUDA 参考程序写回的元素大小。
static size_t output_elem_size(int output_kind) {
    if (output_kind == CAST_OUT_INT32) return sizeof(int32_t);
    if (output_kind == CAST_OUT_INT64) return sizeof(int64_t);
    if (output_kind == CAST_OUT_BOOL) return sizeof(uint8_t);
    return sizeof(float);
}

// 读取输出类型参数；缺省按 float32 reference 输出处理。
static int read_output_kind(const char* params_path) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return -1;
    }
    int output_kind = CAST_OUT_FLOAT32;
    size_t r = fread(&output_kind, sizeof(int), 1, fp);
    fclose(fp);
    if (r != 1) {
        fprintf(stderr, "read params failed\n");
        return -1;
    }
    return output_kind;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];
    int output_kind = read_output_kind(params_path);
    if (output_kind < 0) return 1;

    size_t in_bytes = n * sizeof(float);
    size_t out_bytes = n * output_elem_size(output_kind);
    float* h_input = (float*)malloc(in_bytes);
    void* h_output = malloc(out_bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t r = fread(h_input, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float* d_input = NULL;
    void* d_output = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    cast_kernel<<<blocks, threads>>>(d_input, d_output, n, output_kind);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_output, 1, out_bytes, fo);
    fclose(fo);
    if (w != out_bytes) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
