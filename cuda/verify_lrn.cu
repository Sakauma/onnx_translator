/**
  ******************************************************************************
  * @file        verify_lrn.cu
  * @author      Egor Izmaylov
  * @brief       提供 LRN 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 保存 LRN 参考计算所需的张量形状和 ONNX 属性参数。
struct LrnParams {
    int32_t batch;
    int32_t channels;
    int32_t spatial_size;
    int32_t size;
    float alpha;
    float beta;
    float bias;
};

// 按 ONNX LRN schema 公式计算每个 [N, C, spatial] 位置的归一化结果。
__global__ void lrn_kernel(const double* input, double* output, LrnParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int spatial = (int)(tid % (size_t)p.spatial_size);
    int channel = (int)((tid / (size_t)p.spatial_size) % (size_t)p.channels);
    int batch = (int)(tid / ((size_t)p.channels * (size_t)p.spatial_size));

    int lower = (p.size - 1) / 2;
    int upper = p.size - 1 - lower;
    int begin = channel - lower;
    int end = channel + upper + 1;
    if (begin < 0) begin = 0;
    if (end > p.channels) end = p.channels;

    double square_sum = 0.0;
    for (int c = begin; c < end; c++) {
        size_t idx = ((size_t)batch * (size_t)p.channels + (size_t)c) * (size_t)p.spatial_size + (size_t)spatial;
        double value = input[idx];
        square_sum += value * value;
    }

    double base = (double)p.bias + ((double)p.alpha / (double)p.size) * square_sum;
    output[tid] = input[tid] / pow(base, (double)p.beta);
}

// 从参数文件依次读取 int32 形状参数和 float 属性参数，避免结构体 padding 影响二进制兼容。
static int read_lrn_params(const char* params_path, LrnParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t ints[4];
    float floats[3];
    size_t int_count = fread(ints, sizeof(int32_t), 4, fp);
    size_t float_count = fread(floats, sizeof(float), 3, fp);
    fclose(fp);
    if (int_count != 4 || float_count != 3) {
        fprintf(stderr, "read params failed\n");
        return 0;
    }

    params->batch = ints[0];
    params->channels = ints[1];
    params->spatial_size = ints[2];
    params->size = ints[3];
    params->alpha = floats[0];
    params->beta = floats[1];
    params->bias = floats[2];
    return 1;
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

    LrnParams params;
    if (!read_lrn_params(params_path, &params)) {
        return 1;
    }
    if (params.batch <= 0 || params.channels <= 0 || params.spatial_size <= 0 || params.size <= 0) {
        fprintf(stderr, "invalid params\n");
        return 1;
    }

    size_t input_len = (size_t)params.batch * (size_t)params.channels * (size_t)params.spatial_size;
    if (out_len != input_len) {
        fprintf(stderr, "output length mismatch\n");
        return 1;
    }

    size_t bytes = input_len * sizeof(double);
    double* h_input = (double*)malloc(bytes);
    double* h_output = (double*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }

    FILE* fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }
    size_t read_count = fread(h_input, sizeof(double), input_len, fp);
    fclose(fp);
    if (read_count != input_len) {
        fprintf(stderr, "read input failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }

    double* d_input = NULL;
    double* d_output = NULL;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((input_len + (size_t)threads - 1) / (size_t)threads);
    lrn_kernel<<<blocks, threads>>>(d_input, d_output, params, input_len);
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
    size_t write_count = fwrite(h_output, sizeof(double), input_len, fp);
    fclose(fp);
    if (write_count != input_len) {
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
