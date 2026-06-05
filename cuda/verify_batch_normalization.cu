/**
  ******************************************************************************
  * @file        verify_batch_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 BatchNormalization 算子推理模式的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 保存 BatchNormalization 推理公式所需的 NCHW 展开维度和 epsilon。
struct BatchNormParams {
    int32_t batch;
    int32_t channels;
    int32_t spatial_size;
    float epsilon;
};

// 按 ONNX BatchNormalization 推理公式计算每个输出元素。
__global__ void batch_norm_kernel(
    const double* x,
    const double* scale,
    const double* bias,
    const double* mean,
    const double* var,
    double* out,
    BatchNormParams p,
    size_t total
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int c = (int)((tid / (size_t)p.spatial_size) % (size_t)p.channels);
    double inv_std = 1.0 / sqrt(var[c] + (double)p.epsilon);
    out[tid] = scale[c] * (x[tid] - mean[c]) * inv_std + bias[c];
}

// 从 params.bin 顺序读取 `[N, C, spatial_size] + epsilon`，避免结构体 padding 影响二进制兼容。
static int read_batch_norm_params(const char* params_path, BatchNormParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t dims[3];
    if (fread(dims, sizeof(int32_t), 3, fp) != 3) {
        fprintf(stderr, "read params dims failed\n");
        fclose(fp);
        return 0;
    }
    if (fread(&params->epsilon, sizeof(float), 1, fp) != 1) {
        fprintf(stderr, "read epsilon failed\n");
        fclose(fp);
        return 0;
    }
    fclose(fp);

    params->batch = dims[0];
    params->channels = dims[1];
    params->spatial_size = dims[2];
    return params->batch > 0 && params->channels > 0 && params->spatial_size > 0;
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
    // <out_len> <x.bin> <scale.bin> <bias.bin> <mean.bin> <var.bin> <params.bin> <out.bin>
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <scale.bin> <bias.bin> <mean.bin> <var.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* scale_path = argv[3];
    const char* bias_path = argv[4];
    const char* mean_path = argv[5];
    const char* var_path = argv[6];
    const char* params_path = argv[7];
    const char* out_path = argv[8];

    BatchNormParams params;
    if (!read_batch_norm_params(params_path, &params)) {
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
    double* h_mean = (double*)malloc(param_bytes);
    double* h_var = (double*)malloc(param_bytes);
    double* h_out = (double*)malloc(x_bytes);
    if (!h_x || !h_scale || !h_bias || !h_mean || !h_var || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_mean);
        free(h_var);
        free(h_out);
        return 1;
    }

    if (
        !read_double_array(x_path, h_x, out_len, "x")
        || !read_double_array(scale_path, h_scale, (size_t)params.channels, "scale")
        || !read_double_array(bias_path, h_bias, (size_t)params.channels, "bias")
        || !read_double_array(mean_path, h_mean, (size_t)params.channels, "mean")
        || !read_double_array(var_path, h_var, (size_t)params.channels, "var")
    ) {
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_mean);
        free(h_var);
        free(h_out);
        return 1;
    }

    double* d_x = NULL;
    double* d_scale = NULL;
    double* d_bias = NULL;
    double* d_mean = NULL;
    double* d_var = NULL;
    double* d_out = NULL;
    cudaMalloc((void**)&d_x, x_bytes);
    cudaMalloc((void**)&d_scale, param_bytes);
    cudaMalloc((void**)&d_bias, param_bytes);
    cudaMalloc((void**)&d_mean, param_bytes);
    cudaMalloc((void**)&d_var, param_bytes);
    cudaMalloc((void**)&d_out, x_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_var, h_var, param_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    batch_norm_kernel<<<blocks, threads>>>(d_x, d_scale, d_bias, d_mean, d_var, d_out, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_x);
        cudaFree(d_scale);
        cudaFree(d_bias);
        cudaFree(d_mean);
        cudaFree(d_var);
        cudaFree(d_out);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_mean);
        free(h_var);
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
        cudaFree(d_mean);
        cudaFree(d_var);
        cudaFree(d_out);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_mean);
        free(h_var);
        free(h_out);
        return 1;
    }

    cudaFree(d_x);
    cudaFree(d_scale);
    cudaFree(d_bias);
    cudaFree(d_mean);
    cudaFree(d_var);
    cudaFree(d_out);
    free(h_x);
    free(h_scale);
    free(h_bias);
    free(h_mean);
    free(h_var);
    free(h_out);
    return 0;
}
