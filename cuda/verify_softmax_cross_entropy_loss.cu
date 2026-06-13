/**
  ******************************************************************************
  * @file        verify_softmax_cross_entropy_loss.cu
  * @author      Egor Izmaylov
  * @brief       提供 SoftmaxCrossEntropyLoss 算子的 CUDA 参考验证程序。
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

// 在单个 CUDA 线程中执行 SCE reference，同时可选写回完整 log_prob。
__global__ void sce_loss_kernel(
    const double* scores,
    const int64_t* labels,
    const double* weights,
    double* loss_output,
    double* log_prob_output,
    LossParams params
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double loss_sum = 0.0;
    double denom = 0.0;

    for (int n = 0; n < params.batch; ++n) {
        for (int s = 0; s < params.spatial; ++s) {
            double max_val = -INFINITY;
            for (int c = 0; c < params.classes; ++c) {
                size_t idx = (size_t)n * (size_t)params.classes * (size_t)params.spatial + (size_t)c * (size_t)params.spatial + (size_t)s;
                double value = scores[idx];
                if (value > max_val) max_val = value;
            }

            double exp_sum = 0.0;
            for (int c = 0; c < params.classes; ++c) {
                size_t idx = (size_t)n * (size_t)params.classes * (size_t)params.spatial + (size_t)c * (size_t)params.spatial + (size_t)s;
                exp_sum += exp(scores[idx] - max_val);
            }
            double log_sum = log(exp_sum);

            size_t flat_target = (size_t)n * (size_t)params.spatial + (size_t)s;
            int64_t cls = labels[flat_target];
            double selected_loss = 0.0;
            double cur_weight = 0.0;
            for (int c = 0; c < params.classes; ++c) {
                size_t idx = (size_t)n * (size_t)params.classes * (size_t)params.spatial + (size_t)c * (size_t)params.spatial + (size_t)s;
                double log_prob = scores[idx] - max_val - log_sum;
                if (log_prob_output) log_prob_output[idx] = log_prob;
                if (c == cls && !(params.has_ignore_index && cls == params.ignore_index)) {
                    cur_weight = params.has_weight ? weights[cls] : 1.0;
                    selected_loss = -log_prob * cur_weight;
                }
            }

            if (params.reduction == 0) {
                loss_output[flat_target] = selected_loss;
            } else {
                loss_sum += selected_loss;
                if (!(params.has_ignore_index && cls == params.ignore_index)) {
                    denom += params.has_weight ? cur_weight : 1.0;
                }
            }
        }
    }

    if (params.reduction == 2) {
        loss_output[0] = loss_sum;
    } else if (params.reduction == 1) {
        loss_output[0] = denom == 0.0 ? NAN : loss_sum / denom;
    }
}

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

// 作为 CUDA 验证程序入口，执行 SCE reference，并在需要时写回 log_prob sidecar。
int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <scores.bin> <labels.bin> <weights|null> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    LossParams params;
    if (!read_params(argv[5], &params)) return 1;
    size_t scores_len = (size_t)params.batch * (size_t)params.classes * (size_t)params.spatial;
    size_t labels_len = (size_t)params.batch * (size_t)params.spatial;
    size_t expected_out = params.reduction == 0 ? labels_len : 1;
    if (out_len != expected_out) return 1;

    double* h_scores = (double*)malloc(scores_len * sizeof(double));
    int64_t* h_labels = (int64_t*)malloc(labels_len * sizeof(int64_t));
    double* h_weights = params.has_weight ? (double*)malloc((size_t)params.classes * sizeof(double)) : NULL;
    double* h_loss = (double*)calloc(out_len, sizeof(double));
    double* h_log_prob = params.emit_log_prob ? (double*)malloc(scores_len * sizeof(double)) : NULL;
    if (!h_scores || !h_labels || (params.has_weight && !h_weights) || !h_loss || (params.emit_log_prob && !h_log_prob)) return 1;
    if (!read_double_file(argv[2], h_scores, scores_len)) return 1;
    if (!read_i64_file(argv[3], h_labels, labels_len)) return 1;
    if (params.has_weight && !read_double_file(argv[4], h_weights, (size_t)params.classes)) return 1;

    double* d_scores = NULL;
    int64_t* d_labels = NULL;
    double* d_weights = NULL;
    double* d_loss = NULL;
    double* d_log_prob = NULL;
    cudaMalloc(&d_scores, scores_len * sizeof(double));
    cudaMalloc(&d_labels, labels_len * sizeof(int64_t));
    cudaMalloc(&d_loss, out_len * sizeof(double));
    cudaMemcpy(d_scores, h_scores, scores_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, labels_len * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, out_len * sizeof(double));
    if (params.has_weight) {
        cudaMalloc(&d_weights, (size_t)params.classes * sizeof(double));
        cudaMemcpy(d_weights, h_weights, (size_t)params.classes * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (params.emit_log_prob) {
        cudaMalloc(&d_log_prob, scores_len * sizeof(double));
    }

    sce_loss_kernel<<<1, 1>>>(d_scores, d_labels, d_weights, d_loss, d_log_prob, params);
    cudaDeviceSynchronize();
    cudaMemcpy(h_loss, d_loss, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    if (params.emit_log_prob) {
        cudaMemcpy(h_log_prob, d_log_prob, scores_len * sizeof(double), cudaMemcpyDeviceToHost);
    }
    int ok = write_double_file(argv[6], h_loss, out_len);
    if (params.emit_log_prob) {
        ok = ok && write_double_file("tmp_out_log_prob.bin", h_log_prob, scores_len);
    }

    cudaFree(d_scores);
    cudaFree(d_labels);
    if (d_weights) cudaFree(d_weights);
    cudaFree(d_loss);
    if (d_log_prob) cudaFree(d_log_prob);
    free(h_scores);
    free(h_labels);
    free(h_weights);
    free(h_loss);
    free(h_log_prob);
    return ok ? 0 : 1;
}
