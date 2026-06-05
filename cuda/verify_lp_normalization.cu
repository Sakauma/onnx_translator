/**
  ******************************************************************************
  * @file        verify_lp_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 LpNormalization 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 保存 LpNormalization 参考计算所需的 axis 展开布局和 p 参数。
struct LpNormParams {
    int32_t outer;
    int32_t inner;
    int32_t remaining;
    int32_t p;
};

// 按 ONNX LpNormalization 公式沿指定 axis 计算 x / ||x||_p。
__global__ void lp_norm_kernel(const double* input, double* output, LpNormParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int rem = (int)(tid % (size_t)p.remaining);
    int outer_idx = (int)(tid / ((size_t)p.inner * (size_t)p.remaining));
    size_t base = (size_t)outer_idx * (size_t)p.inner * (size_t)p.remaining + (size_t)rem;

    double sum = 0.0;
    for (int j = 0; j < p.inner; j++) {
        double v = input[base + (size_t)j * (size_t)p.remaining];
        sum += pow(fabs(v), (double)p.p);
    }
    double norm = pow(sum, 1.0 / (double)p.p);
    output[tid] = norm == 0.0 ? 0.0 : input[tid] / norm;
}

// 顺序读取 `[outer, inner, remaining, p]`，避免结构体 padding 影响二进制兼容。
static int read_lp_norm_params(const char* params_path, LpNormParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t values[4];
    if (fread(values, sizeof(int32_t), 4, fp) != 4) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 0;
    }
    fclose(fp);
    params->outer = values[0];
    params->inner = values[1];
    params->remaining = values[2];
    params->p = values[3];
    return params->outer > 0 && params->inner > 0 && params->remaining > 0 && params->p > 0;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    LpNormParams params;
    if (!read_lp_norm_params(params_path, &params)) {
        return 1;
    }
    size_t expected_len = (size_t)params.outer * (size_t)params.inner * (size_t)params.remaining;
    if (out_len != expected_len) {
        fprintf(stderr, "output length mismatch\n");
        return 1;
    }

    size_t bytes = out_len * sizeof(double);
    double* h_input = (double*)malloc(bytes);
    double* h_output = (double*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "host alloc failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }

    FILE* fp = fopen(input_path, "rb");
    if (!fp || fread(h_input, sizeof(double), out_len, fp) != out_len) {
        fprintf(stderr, "read input failed\n");
        if (fp) fclose(fp);
        free(h_input);
        free(h_output);
        return 1;
    }
    fclose(fp);

    double* d_input = NULL;
    double* d_output = NULL;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    lp_norm_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return 1;
    }
    size_t write_count = fwrite(h_output, sizeof(double), out_len, fp);
    fclose(fp);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
