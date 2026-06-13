/**
  ******************************************************************************
  * @file        verify_layer_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 LayerNormalization 算子单输出和 mean/inv_std 多输出路径的 CUDA 参考验证程序。
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

// 保存 LayerNormalization 参考计算所需的分段长度、可选输入标记和 epsilon。
struct LayerNormParams {
    int32_t row_count;
    int32_t normalized_size;
    int32_t has_scale;
    int32_t has_bias;
    int32_t emit_stats;
    float epsilon;
};

// 按 ONNX LayerNormalization 公式对每一行的归一化后缀执行归一化。
__global__ void layer_norm_kernel(
    const double* x,
    const double* scale,
    const double* bias,
    double* out,
    double* mean_out,
    double* inv_std_out,
    LayerNormParams p,
    size_t total
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int row = (int)(tid / (size_t)p.normalized_size);
    int col = (int)(tid % (size_t)p.normalized_size);
    size_t base = (size_t)row * (size_t)p.normalized_size;

    double sum = 0.0;
    for (int i = 0; i < p.normalized_size; i++) {
        sum += x[base + (size_t)i];
    }
    double mean = sum / (double)p.normalized_size;

    double square_sum = 0.0;
    for (int i = 0; i < p.normalized_size; i++) {
        double diff = x[base + (size_t)i] - mean;
        square_sum += diff * diff;
    }
    double variance = square_sum / (double)p.normalized_size;
    double inv_std = 1.0 / sqrt(variance + (double)p.epsilon);
    if (p.emit_stats && col == 0) {
        mean_out[row] = mean;
        inv_std_out[row] = inv_std;
    }

    double y = (x[tid] - mean) * inv_std;
    if (p.has_scale) y *= scale[col];
    if (p.has_bias) y += bias[col];
    out[tid] = y;
}

// 顺序读取 `[row_count, normalized_size, has_scale, has_bias, emit_stats] + epsilon`，避免结构体 padding 影响二进制兼容。
static int read_layer_norm_params(const char* params_path, LayerNormParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t ints[5];
    if (fread(ints, sizeof(int32_t), 5, fp) != 5) {
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

    params->row_count = ints[0];
    params->normalized_size = ints[1];
    params->has_scale = ints[2];
    params->has_bias = ints[3];
    params->emit_stats = ints[4];
    return params->row_count > 0 && params->normalized_size > 0;
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

// 写出 double 二进制数组，供多输出 sidecar 复用。
static int write_double_array(const char* path, const double* data, size_t count, const char* label) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open %s output failed\n", label);
        return 0;
    }
    size_t write_count = fwrite(data, sizeof(double), count, fp);
    fclose(fp);
    if (write_count != count) {
        fprintf(stderr, "write %s output failed\n", label);
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

    LayerNormParams params;
    if (!read_layer_norm_params(params_path, &params)) {
        return 1;
    }

    size_t expected_len = (size_t)params.row_count * (size_t)params.normalized_size;
    if (out_len != expected_len) {
        fprintf(stderr, "output length mismatch\n");
        return 1;
    }

    size_t x_bytes = out_len * sizeof(double);
    size_t param_bytes = (size_t)params.normalized_size * sizeof(double);
    size_t stats_bytes = (size_t)params.row_count * sizeof(double);
    double* h_x = (double*)malloc(x_bytes);
    double* h_scale = (double*)malloc(param_bytes);
    double* h_bias = (double*)malloc(param_bytes);
    double* h_out = (double*)malloc(x_bytes);
    double* h_mean = (double*)malloc(stats_bytes);
    double* h_inv_std = (double*)malloc(stats_bytes);
    if (!h_x || !h_scale || !h_bias || !h_out || !h_mean || !h_inv_std) {
        fprintf(stderr, "host alloc failed\n");
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        free(h_mean);
        free(h_inv_std);
        return 1;
    }

    if (
        !read_double_array(x_path, h_x, out_len, "x")
        || (params.has_scale && !read_double_array(scale_path, h_scale, (size_t)params.normalized_size, "scale"))
        || (params.has_bias && !read_double_array(bias_path, h_bias, (size_t)params.normalized_size, "bias"))
    ) {
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        free(h_mean);
        free(h_inv_std);
        return 1;
    }

    double* d_x = NULL;
    double* d_scale = NULL;
    double* d_bias = NULL;
    double* d_out = NULL;
    double* d_mean = NULL;
    double* d_inv_std = NULL;
    cudaMalloc((void**)&d_x, x_bytes);
    cudaMalloc((void**)&d_scale, param_bytes);
    cudaMalloc((void**)&d_bias, param_bytes);
    cudaMalloc((void**)&d_out, x_bytes);
    cudaMalloc((void**)&d_mean, stats_bytes);
    cudaMalloc((void**)&d_inv_std, stats_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, param_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    layer_norm_kernel<<<blocks, threads>>>(d_x, d_scale, d_bias, d_out, d_mean, d_inv_std, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);
    if (params.emit_stats) {
        cudaMemcpy(h_mean, d_mean, stats_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_inv_std, d_inv_std, stats_bytes, cudaMemcpyDeviceToHost);
    }

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_x);
        cudaFree(d_scale);
        cudaFree(d_bias);
        cudaFree(d_out);
        cudaFree(d_mean);
        cudaFree(d_inv_std);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        free(h_mean);
        free(h_inv_std);
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
        cudaFree(d_mean);
        cudaFree(d_inv_std);
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        free(h_mean);
        free(h_inv_std);
        return 1;
    }

    if (params.emit_stats) {
        int sidecar_ok = write_double_array("tmp_layer_norm_mean.bin", h_mean, (size_t)params.row_count, "mean");
        sidecar_ok = sidecar_ok && write_double_array("tmp_layer_norm_inv_std.bin", h_inv_std, (size_t)params.row_count, "inv_std");
        if (!sidecar_ok) {
            cudaFree(d_x);
            cudaFree(d_scale);
            cudaFree(d_bias);
            cudaFree(d_out);
            cudaFree(d_mean);
            cudaFree(d_inv_std);
            free(h_x);
            free(h_scale);
            free(h_bias);
            free(h_out);
            free(h_mean);
            free(h_inv_std);
            return 1;
        }
    }

    cudaFree(d_x);
    cudaFree(d_scale);
    cudaFree(d_bias);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_inv_std);
    free(h_x);
    free(h_scale);
    free(h_bias);
    free(h_out);
    free(h_mean);
    free(h_inv_std);
    return 0;
}
