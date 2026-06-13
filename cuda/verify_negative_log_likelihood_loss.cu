/**
  ******************************************************************************
  * @file        verify_negative_log_likelihood_loss.cu
  * @author      Egor Izmaylov
  * @brief       提供 NegativeLogLikelihoodLoss 算子的 CUDA 参考验证程序。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct LossParams {
    int32_t batch;
    int32_t classes;
    int32_t spatial;
    int32_t reduction;
    int32_t has_weight;
    int32_t has_ignore_index;
    int32_t emit_log_prob;
    int64_t ignore_index;
};

// 在单个 CUDA 线程中执行 NLLLoss reference，避免 reduction 顺序差异影响 C/CUDA 对齐。
__global__ void nll_loss_kernel(
    const double* input,
    const int64_t* target,
    const double* weight,
    double* output,
    LossParams params
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    size_t total = (size_t)params.batch * (size_t)params.spatial;
    double sum = 0.0;
    double denom = 0.0;

    for (size_t i = 0; i < total; ++i) {
        int64_t cls = target[i];
        double cur_weight = 0.0;
        double weighted_loss = 0.0;
        if (!(params.has_ignore_index && cls == params.ignore_index) && cls >= 0 && cls < params.classes) {
            cur_weight = params.has_weight ? weight[cls] : 1.0;
            size_t n = i / (size_t)params.spatial;
            size_t s = i % (size_t)params.spatial;
            size_t input_idx = n * (size_t)params.classes * (size_t)params.spatial + (size_t)cls * (size_t)params.spatial + s;
            weighted_loss = -input[input_idx] * cur_weight;
        }

        if (params.reduction == 0) {
            output[i] = weighted_loss;
        } else {
            sum += weighted_loss;
            denom += (params.has_weight || params.has_ignore_index) ? cur_weight : 1.0;
        }
    }

    if (params.reduction == 2) {
        output[0] = sum;
    } else if (params.reduction == 1) {
        output[0] = denom == 0.0 ? NAN : sum / denom;
    }
}

// 按固定二进制布局读取 loss 参数，避免结构体 padding 影响兼容性。
static int read_params(const char* path, LossParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int32_t ints[7];
    if (fread(ints, sizeof(int32_t), 7, fp) != 7) {
        fclose(fp);
        return 0;
    }
    int64_t ignore_index = 0;
    if (fread(&ignore_index, sizeof(int64_t), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    params->batch = ints[0];
    params->classes = ints[1];
    params->spatial = ints[2];
    params->reduction = ints[3];
    params->has_weight = ints[4];
    params->has_ignore_index = ints[5];
    params->emit_log_prob = ints[6];
    params->ignore_index = ignore_index;
    return params->batch > 0 && params->classes > 0 && params->spatial > 0;
}

static int read_double_file(const char* path, double* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t got = fread(data, sizeof(double), n, fp);
    fclose(fp);
    return got == n;
}

static int read_i64_file(const char* path, int64_t* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t got = fread(data, sizeof(int64_t), n, fp);
    fclose(fp);
    return got == n;
}

static int write_double_file(const char* path, const double* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(double), n, fp);
    fclose(fp);
    return wrote == n;
}

// 作为 CUDA 验证程序入口，执行 NLLLoss reference 并写回 loss 输出。
int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <target.bin> <weight|null> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    LossParams params;
    if (!read_params(argv[5], &params)) return 1;
    size_t input_len = (size_t)params.batch * (size_t)params.classes * (size_t)params.spatial;
    size_t target_len = (size_t)params.batch * (size_t)params.spatial;
    size_t expected_out = params.reduction == 0 ? target_len : 1;
    if (out_len != expected_out) return 1;

    double* h_input = (double*)malloc(input_len * sizeof(double));
    int64_t* h_target = (int64_t*)malloc(target_len * sizeof(int64_t));
    double* h_weight = params.has_weight ? (double*)malloc((size_t)params.classes * sizeof(double)) : NULL;
    double* h_output = (double*)calloc(out_len, sizeof(double));
    if (!h_input || !h_target || (params.has_weight && !h_weight) || !h_output) return 1;
    if (!read_double_file(argv[2], h_input, input_len)) return 1;
    if (!read_i64_file(argv[3], h_target, target_len)) return 1;
    if (params.has_weight && !read_double_file(argv[4], h_weight, (size_t)params.classes)) return 1;

    double* d_input = NULL;
    int64_t* d_target = NULL;
    double* d_weight = NULL;
    double* d_output = NULL;
    cudaMalloc(&d_input, input_len * sizeof(double));
    cudaMalloc(&d_target, target_len * sizeof(int64_t));
    cudaMalloc(&d_output, out_len * sizeof(double));
    cudaMemcpy(d_input, h_input, input_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, target_len * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_len * sizeof(double));
    if (params.has_weight) {
        cudaMalloc(&d_weight, (size_t)params.classes * sizeof(double));
        cudaMemcpy(d_weight, h_weight, (size_t)params.classes * sizeof(double), cudaMemcpyHostToDevice);
    }

    nll_loss_kernel<<<1, 1>>>(d_input, d_target, d_weight, d_output, params);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    int ok = write_double_file(argv[6], h_output, out_len);

    cudaFree(d_input);
    cudaFree(d_target);
    if (d_weight) cudaFree(d_weight);
    cudaFree(d_output);
    free(h_input);
    free(h_target);
    free(h_weight);
    free(h_output);
    return ok ? 0 : 1;
}
