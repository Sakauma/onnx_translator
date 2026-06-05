/**
  ******************************************************************************
  * @file        verify_isinf.cu
  * @author      Egor Izmaylov
  * @brief       提供 IsInf 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 按 detect_positive 和 detect_negative 标志判断正负无穷。
__global__ void isinf_kernel(const float* input, unsigned char* output, int detect_pos, int detect_neg, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n) return;

    float value = input[idx];
    unsigned char result = 0;
    if (isinf(value)) {
        if (value > 0.0f && detect_pos) {
            result = 1;
        } else if (value < 0.0f && detect_neg) {
            result = 1;
        }
    }
    output[idx] = result;
}

// 读取 `[detect_positive, detect_negative]` 参数。
static int read_isinf_params(const char* params_path, int* detect_pos, int* detect_neg) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t values[2];
    if (fread(values, sizeof(int32_t), 2, fp) != 2) {
        fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    fclose(fp);
    *detect_pos = values[0] != 0;
    *detect_neg = values[1] != 0;
    return 1;
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
    size_t in_bytes = n * sizeof(float);
    size_t out_bytes = n * sizeof(unsigned char);

    int detect_pos = 1;
    int detect_neg = 1;
    if (!read_isinf_params(params_path, &detect_pos, &detect_neg)) {
        return 1;
    }

    float* h_input = (float*)malloc(in_bytes);
    unsigned char* h_output = (unsigned char*)malloc(out_bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), n, fi);
    fclose(fi);
    if (ri != n) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_input = NULL;
    unsigned char* d_output = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    isinf_kernel<<<blocks, threads>>>(d_input, d_output, detect_pos, detect_neg, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(unsigned char), n, fo);
    fclose(fo);
    if (wo != n) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
