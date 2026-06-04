/**
  ******************************************************************************
  * @file        verify_sum.cu
  * @author      Egor Izmaylov
  * @brief       提供 sum 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Sum 对多个已经按 ONNX 广播规则展开的输入逐元素求和。
__global__ void sum_kernel(const float* inputs, float* output, size_t n, int num_inputs) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= n) return;

    double acc = 0.0;
    for (int i = 0; i < num_inputs; ++i) {
        acc += (double)inputs[(size_t)i * n + t];
    }
    output[t] = (float)acc;
}

// 读取一个 float32 输入文件到连续输入缓冲区中的指定分片。
static int read_input_slice(const char* path, float* dst, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed: %s\n", path);
        return 0;
    }
    size_t r = fread(dst, sizeof(float), n, fp);
    fclose(fp);
    if (r != n) {
        fprintf(stderr, "fread mismatch: %s\n", path);
        return 0;
    }
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input0.bin> ... <inputN.bin> <out.bin>
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <out_len> <input0.bin> ... <inputN.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    int num_inputs = argc - 3;
    const char* out_path = argv[argc - 1];
    size_t bytes = n * sizeof(float);
    size_t all_bytes = (size_t)num_inputs * bytes;

    float* h_inputs = (float*)malloc(all_bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_inputs || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < num_inputs; ++i) {
        if (!read_input_slice(argv[2 + i], h_inputs + (size_t)i * n, n)) {
            free(h_inputs);
            free(h_output);
            return 1;
        }
    }

    float* d_inputs = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_inputs, all_bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_inputs, h_inputs, all_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    sum_kernel<<<blocks, threads>>>(d_inputs, d_output, n, num_inputs);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
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

    cudaFree(d_inputs);
    cudaFree(d_output);
    free(h_inputs);
    free(h_output);
    return 0;
}
