/**
  ******************************************************************************
  * @file        verify_selu.cu
  * @author      Egor Izmaylov
  * @brief       提供 selu 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Selu 按 ONNX 定义逐元素计算 gamma * (x > 0 ? x : alpha * (exp(x) - 1))。
__global__ void selu_kernel(const float* input, float* output, size_t n, float alpha, float gamma) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < n) {
        float x = input[t];
        float inner = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
        output[t] = gamma * inner;
    }
}

// 从参数文件读取 alpha/gamma；调用方始终按两个 float32 写入。
static int read_params(const char* params_path, float* alpha, float* gamma) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    float params[2];
    size_t r = fread(params, sizeof(float), 2, fp);
    fclose(fp);
    if (r != 2) {
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    *alpha = params[0];
    *gamma = params[1];
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
    float alpha = 1.67326f;
    float gamma = 1.0507f;
    if (!read_params(argv[3], &alpha, &gamma)) {
        return 1;
    }

    size_t bytes = n * sizeof(float);
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(argv[2], "rb");
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
    float* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    selu_kernel<<<blocks, threads>>>(d_input, d_output, n, alpha, gamma);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[4], "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_output, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
