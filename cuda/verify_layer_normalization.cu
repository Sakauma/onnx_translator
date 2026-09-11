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
#include <cuda_fp16.h>
#include "verify_common.cuh"
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
    int32_t stash_type;
    int32_t input_dtype;
    float epsilon;
};

// 将 float32 按 round-to-nearest-even 物化为 bfloat16，再解码回 float32。
__device__ static float materialize_bfloat16(float value) {
    uint32_t bits = __float_as_uint(value);
    uint32_t exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return __uint_as_float(bits & 0xffff0000u);
    }
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(rounded & 0xffff0000u);
}

__device__ static float stash_value(float value, int32_t stash_type) {
    return stash_type == 16 ? materialize_bfloat16(value) : value;
}

// 按输入类型 T 物化 stage-two 的 CastLike、Mul 和 Add。
__device__ static float materialize_float16(float value) {
    return __half2float(__float2half_rn(value));
}

__device__ static float materialize_input_float(float value, int32_t input_dtype) {
    if (input_dtype == 10) return materialize_float16(value);
    if (input_dtype == 16) return materialize_bfloat16(value);
    return value;
}

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

    float sum = stash_value(0.0f, p.stash_type);
    for (int i = 0; i < p.normalized_size; i++) {
        float value = stash_value((float)x[base + (size_t)i], p.stash_type);
        sum = stash_value(sum + value, p.stash_type);
    }
    float divisor = stash_value((float)p.normalized_size, p.stash_type);
    float mean = stash_value(sum / divisor, p.stash_type);

    float square_sum = stash_value(0.0f, p.stash_type);
    for (int i = 0; i < p.normalized_size; i++) {
        float value = stash_value((float)x[base + (size_t)i], p.stash_type);
        float diff = stash_value(value - mean, p.stash_type);
        float square = stash_value(diff * diff, p.stash_type);
        square_sum = stash_value(square_sum + square, p.stash_type);
    }
    float variance = stash_value(square_sum / divisor, p.stash_type);
    float epsilon = stash_value(p.epsilon, p.stash_type);
    float variance_with_epsilon = stash_value(variance + epsilon, p.stash_type);
    float standard_deviation = stash_value(sqrtf(variance_with_epsilon), p.stash_type);
    float inv_std = stash_value(stash_value(1.0f, p.stash_type) / standard_deviation, p.stash_type);
    if (p.emit_stats && col == 0) {
        mean_out[row] = mean;
        inv_std_out[row] = inv_std;
    }

    float value = stash_value((float)x[tid], p.stash_type);
    float normalized = stash_value(stash_value(value - mean, p.stash_type) * inv_std, p.stash_type);
    if (p.input_dtype == 11) {
        double y = (double)normalized;
        if (p.has_scale) y = y * scale[col];
        if (p.has_bias) y = y + bias[col];
        out[tid] = y;
    } else {
        float y = materialize_input_float(normalized, p.input_dtype);
        if (p.has_scale) {
            float scale_value = materialize_input_float((float)scale[col], p.input_dtype);
            y = materialize_input_float(y * scale_value, p.input_dtype);
        }
        if (p.has_bias) {
            float bias_value = materialize_input_float((float)bias[col], p.input_dtype);
            y = materialize_input_float(y + bias_value, p.input_dtype);
        }
        out[tid] = (double)y;
    }
}

// 顺序读取 `[row_count, normalized_size, has_scale, has_bias, emit_stats, stash_type, input_dtype] + epsilon`，避免结构体 padding 影响二进制兼容。
static int read_layer_norm_params(const char* params_path, LayerNormParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t ints[7];
    if (fread(ints, sizeof(int32_t), 7, fp) != 7) {
        fprintf(stderr, "read params ints failed\n");
        verify_close_file(fp);
        return 0;
    }
    if (fread(&params->epsilon, sizeof(float), 1, fp) != 1) {
        fprintf(stderr, "read epsilon failed\n");
        verify_close_file(fp);
        return 0;
    }
    verify_close_file(fp);

    params->row_count = ints[0];
    params->normalized_size = ints[1];
    params->has_scale = ints[2];
    params->has_bias = ints[3];
    params->emit_stats = ints[4];
    params->stash_type = ints[5];
    params->input_dtype = ints[6];
    if (params->stash_type != 1 && params->stash_type != 16) {
        fprintf(stderr, "unsupported stash_type\n");
        return 0;
    }
    if (params->input_dtype != 1 && params->input_dtype != 10
        && params->input_dtype != 11 && params->input_dtype != 16) {
        fprintf(stderr, "unsupported input_dtype\n");
        return 0;
    }
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
    verify_close_file(fp);
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
    verify_close_file(fp);
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
    CUDA_CHECK(cudaMalloc((void**)&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_scale, param_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_bias, param_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, x_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_mean, stats_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inv_std, stats_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale, param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, param_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    layer_norm_kernel<<<blocks, threads>>>(d_x, d_scale, d_bias, d_out, d_mean, d_inv_std, params, out_len);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost));
    if (params.emit_stats) {
        CUDA_CHECK(cudaMemcpy(h_mean, d_mean, stats_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_inv_std, d_inv_std, stats_bytes, cudaMemcpyDeviceToHost));
    }

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_scale));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_mean));
        CUDA_CHECK(cudaFree(d_inv_std));
        free(h_x);
        free(h_scale);
        free(h_bias);
        free(h_out);
        free(h_mean);
        free(h_inv_std);
        return 1;
    }
    size_t write_count = fwrite(h_out, sizeof(double), out_len, fp);
    verify_close_file(fp);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_scale));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_mean));
        CUDA_CHECK(cudaFree(d_inv_std));
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
            CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_scale));
            CUDA_CHECK(cudaFree(d_bias));
            CUDA_CHECK(cudaFree(d_out));
            CUDA_CHECK(cudaFree(d_mean));
            CUDA_CHECK(cudaFree(d_inv_std));
            free(h_x);
            free(h_scale);
            free(h_bias);
            free(h_out);
            free(h_mean);
            free(h_inv_std);
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_inv_std));
    free(h_x);
    free(h_scale);
    free(h_bias);
    free(h_out);
    free(h_mean);
    free(h_inv_std);
    return 0;
}
