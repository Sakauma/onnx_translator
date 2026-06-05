/**
  ******************************************************************************
  * @file        verify_group_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 GroupNormalization 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 保存 GroupNormalization 参考计算所需的 NCHW 展开维度、分组数和 epsilon。
struct GroupNormParams {
    int32_t batch;
    int32_t channels;
    int32_t spatial_size;
    int32_t num_groups;
    float epsilon;
};

// 按 ONNX GroupNormalization 公式对每个 batch/group 计算均值方差并应用通道仿射。
__global__ void group_norm_kernel(
    const double* x,
    const double* scale,
    const double* bias,
    double* out,
    GroupNormParams p,
    size_t total
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int spatial = (int)(tid % (size_t)p.spatial_size);
    int c = (int)((tid / (size_t)p.spatial_size) % (size_t)p.channels);
    int n = (int)(tid / ((size_t)p.channels * (size_t)p.spatial_size));
    int channels_per_group = p.channels / p.num_groups;
    int group = c / channels_per_group;
    int c_start = group * channels_per_group;
    int c_end = c_start + channels_per_group;
    size_t batch_offset = (size_t)n * (size_t)p.channels * (size_t)p.spatial_size;
    size_t group_size = (size_t)channels_per_group * (size_t)p.spatial_size;

    double sum = 0.0;
    for (int gc = c_start; gc < c_end; gc++) {
        size_t offset = batch_offset + (size_t)gc * (size_t)p.spatial_size;
        for (int i = 0; i < p.spatial_size; i++) {
            sum += x[offset + (size_t)i];
        }
    }
    double mean = sum / (double)group_size;

    double square_sum = 0.0;
    for (int gc = c_start; gc < c_end; gc++) {
        size_t offset = batch_offset + (size_t)gc * (size_t)p.spatial_size;
        for (int i = 0; i < p.spatial_size; i++) {
            double diff = x[offset + (size_t)i] - mean;
            square_sum += diff * diff;
        }
    }
    double variance = square_sum / (double)group_size;
    double inv_std = 1.0 / sqrt(variance + (double)p.epsilon);
    size_t idx = batch_offset + (size_t)c * (size_t)p.spatial_size + (size_t)spatial;
    out[tid] = (x[idx] - mean) * inv_std * scale[c] + bias[c];
}

// 顺序读取 `[N, C, spatial_size, num_groups] + epsilon`，避免结构体 padding 影响二进制兼容。
static int read_group_norm_params(const char* params_path, GroupNormParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t ints[4];
    if (fread(ints, sizeof(int32_t), 4, fp) != 4) {
        fprintf(stderr, "read params ints failed\n");
        fclose(fp);
        return 0;
    }
    if (fread(&params->epsilon, sizeof(float), 1, fp) != 1) {
        fprintf(stderr, "read epsilon failed\n");
        fclose(fp);
        return 0;
    }
    fclose(fp);

    params->batch = ints[0];
    params->channels = ints[1];
    params->spatial_size = ints[2];
    params->num_groups = ints[3];
    return (
        params->batch > 0
        && params->channels > 0
        && params->spatial_size > 0
        && params->num_groups > 0
        && params->channels % params->num_groups == 0
    );
}

// 读取 double 二进制数组，统一处理文件打开、长度校验和错误信息。
static int read_double_array(const char* path, double* data, size_t count, const char* label) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s failed\n", label);
        return 0;
    }
    size_t read_count = fread(data, sizeof(double), count, fp);
    fclose(fp);
    if (read_count != count) {
        fprintf(stderr, "read %s failed\n", label);
        return 0;
    }
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <x.bin> <scale.bin> <bias.bin> <params.bin> <out.bin>
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <scale.bin> <bias.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* scale_path = argv[3];
    const char* bias_path = argv[4];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    GroupNormParams params;
    if (!read_group_norm_params(params_path, &params)) {
        return 1;
    }
    size_t expected_len = (size_t)params.batch * (size_t)params.channels * (size_t)params.spatial_size;
    if (out_len != expected_len) {
        fprintf(stderr, "output length mismatch\n");
        return 1;
    }

    size_t x_bytes = out_len * sizeof(double);
    size_t param_bytes = (size_t)params.channels * sizeof(double);
    double* h_x = (double*)malloc(x_bytes);
    double* h_scale = (double*)malloc(param_bytes);
    double* h_bias = (double*)malloc(param_bytes);
    double* h_out = (double*)malloc(x_bytes);
    if (!h_x || !h_scale || !h_bias || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        return 1;
    }

    if (
        !read_double_array(x_path, h_x, out_len, "x")
        || !read_double_array(scale_path, h_scale, (size_t)params.channels, "scale")
        || !read_double_array(bias_path, h_bias, (size_t)params.channels, "bias")
    ) {
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        return 1;
    }

    double* d_x = NULL;
    double* d_scale = NULL;
    double* d_bias = NULL;
    double* d_out = NULL;
    cudaMalloc((void**)&d_x, x_bytes);
    cudaMalloc((void**)&d_scale, param_bytes);
    cudaMalloc((void**)&d_bias, param_bytes);
    cudaMalloc((void**)&d_out, x_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, param_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    group_norm_kernel<<<blocks, threads>>>(d_x, d_scale, d_bias, d_out, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_x);
        cudaFree(d_scale);
        cudaFree(d_bias);
        cudaFree(d_out);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        return 1;
    }
    size_t write_count = fwrite(h_out, sizeof(double), out_len, fp);
    fclose(fp);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        cudaFree(d_x);
        cudaFree(d_scale);
        cudaFree(d_bias);
        cudaFree(d_out);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        return 1;
    }

    cudaFree(d_x);
    cudaFree(d_scale);
    cudaFree(d_bias);
    cudaFree(d_out);
    free(h_x);
    free(h_scale);
    free(h_bias);
    free(h_out);
    return 0;
}
