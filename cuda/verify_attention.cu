/**
  ******************************************************************************
  * @file        verify_attention.cu
  * @author      Egor Izmaylov
  * @brief       提供 Attention 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

struct AttentionParams {
    int32_t batch_size;
    int32_t q_heads;
    int32_t kv_heads;
    int32_t q_seq;
    int32_t kv_seq;
    int32_t head_size;
    int32_t v_head_size;
    int32_t has_mask;
    int32_t mask_is_bool;
    int32_t mask_rank;
    int32_t mask_shape0;
    int32_t mask_shape1;
    int32_t mask_shape2;
    int32_t mask_shape3;
    int32_t is_causal;
    float scale;
    float softcap;
};

// 根据右对齐广播规则读取 Attention mask 的线性下标。
__device__ int attention_mask_index(AttentionParams p, int batch, int head, int query, int key, size_t* index) {
    if (!index || p.mask_rank <= 0 || p.mask_rank > 4) return 0;
    int shapes[4] = {p.mask_shape0, p.mask_shape1, p.mask_shape2, p.mask_shape3};
    int target[4] = {batch, head, query, key};
    int offset = 4 - p.mask_rank;
    size_t flat = 0;
    size_t stride = 1;
    for (int dim = p.mask_rank - 1; dim >= 0; --dim) {
        int coord = target[dim + offset];
        int size = shapes[dim];
        if (size == 1) {
            coord = 0;
        } else if (coord >= size) {
            return 0;
        }
        flat += (size_t)coord * stride;
        stride *= (size_t)size;
    }
    *index = flat;
    return 1;
}

// 计算单个 QK 分数，并应用 mask、causal 和 softcap。
__device__ double attention_score(const double* q, const double* k, const double* mask,
                                  AttentionParams p, int batch, int q_head, int kv_head,
                                  int query, int key) {
    double score = 0.0;
    for (int dim = 0; dim < p.head_size; ++dim) {
        size_t q_idx = (((size_t)batch * (size_t)p.q_heads + (size_t)q_head) * (size_t)p.q_seq + (size_t)query)
                     * (size_t)p.head_size + (size_t)dim;
        size_t k_idx = (((size_t)batch * (size_t)p.kv_heads + (size_t)kv_head) * (size_t)p.kv_seq + (size_t)key)
                     * (size_t)p.head_size + (size_t)dim;
        score += q[q_idx] * k[k_idx];
    }
    double scale = p.scale >= 0.0f ? (double)p.scale : 1.0 / sqrt((double)p.head_size);
    score *= scale;

    if (p.has_mask) {
        size_t mask_idx = 0;
        if (attention_mask_index(p, batch, q_head, query, key, &mask_idx)) {
            double mask_value = mask[mask_idx];
            if (p.mask_is_bool) {
                if (mask_value == 0.0) score = -INFINITY;
            } else {
                score += mask_value;
            }
        } else {
            score = -INFINITY;
        }
    }
    if (p.is_causal && key > query) {
        score = -INFINITY;
    }
    if (p.softcap > 0.0f) {
        score = tanh(score / (double)p.softcap) * (double)p.softcap;
    }
    return score;
}

// 执行 Attention CUDA reference kernel，每个线程负责一个输出元素。
__global__ void attention_kernel(const double* q, const double* k, const double* v, const double* mask,
                                 double* y, AttentionParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int value_dim = (int)(tid % (size_t)p.v_head_size);
    size_t tmp = tid / (size_t)p.v_head_size;
    int query = (int)(tmp % (size_t)p.q_seq); tmp /= (size_t)p.q_seq;
    int q_head = (int)(tmp % (size_t)p.q_heads); tmp /= (size_t)p.q_heads;
    int batch = (int)tmp;
    int head_repeat = p.q_heads / p.kv_heads;
    int kv_head = q_head / head_repeat;

    double max_score = -DBL_MAX;
    for (int key = 0; key < p.kv_seq; ++key) {
        double score = attention_score(q, k, mask, p, batch, q_head, kv_head, query, key);
        if (score > max_score) max_score = score;
    }

    double denom = 0.0;
    for (int key = 0; key < p.kv_seq; ++key) {
        double score = attention_score(q, k, mask, p, batch, q_head, kv_head, query, key);
        denom += exp(score - max_score);
    }

    double sum = 0.0;
    for (int key = 0; key < p.kv_seq; ++key) {
        double score = attention_score(q, k, mask, p, batch, q_head, kv_head, query, key);
        double weight = exp(score - max_score) / denom;
        size_t v_idx = (((size_t)batch * (size_t)p.kv_heads + (size_t)kv_head) * (size_t)p.kv_seq + (size_t)key)
                     * (size_t)p.v_head_size + (size_t)value_dim;
        sum += weight * v[v_idx];
    }
    y[tid] = sum;
}

// 读取一个二进制文件到指定 vector 中。
template <typename T>
static int read_vector(const char* path, std::vector<T>& data) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t count = fread(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
    return count == data.size();
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <q.bin> <k.bin> <v.bin> [mask|null] <params.bin> <out.bin>
    if (argc != 7 && argc != 8) {
        fprintf(stderr, "Usage: %s <out_len> <q.bin> <k.bin> <v.bin> [mask|null] <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t out_len = (size_t)atoll(argv[1]);
    const char* q_path = argv[2];
    const char* k_path = argv[3];
    const char* v_path = argv[4];
    const char* mask_path = argc == 8 ? argv[5] : "null";
    const char* params_path = argc == 8 ? argv[6] : argv[5];
    const char* out_path = argc == 8 ? argv[7] : argv[6];

    AttentionParams p;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) return 2;
    if (fread(&p, sizeof(AttentionParams), 1, fp) != 1) {
        fclose(fp);
        return 3;
    }
    fclose(fp);

    size_t expected = (size_t)p.batch_size * (size_t)p.q_heads * (size_t)p.q_seq * (size_t)p.v_head_size;
    if (out_len != expected || p.q_heads <= 0 || p.kv_heads <= 0 || p.q_heads % p.kv_heads != 0) return 4;
    size_t q_len = (size_t)p.batch_size * (size_t)p.q_heads * (size_t)p.q_seq * (size_t)p.head_size;
    size_t k_len = (size_t)p.batch_size * (size_t)p.kv_heads * (size_t)p.kv_seq * (size_t)p.head_size;
    size_t v_len = (size_t)p.batch_size * (size_t)p.kv_heads * (size_t)p.kv_seq * (size_t)p.v_head_size;
    int mask_shapes[4] = {p.mask_shape0, p.mask_shape1, p.mask_shape2, p.mask_shape3};
    size_t mask_len = 0;
    if (p.has_mask) {
        mask_len = 1;
        for (int i = 0; i < p.mask_rank; ++i) mask_len *= (size_t)mask_shapes[i];
    }

    std::vector<double> h_q(q_len);
    std::vector<double> h_k(k_len);
    std::vector<double> h_v(v_len);
    std::vector<double> h_mask(mask_len);
    std::vector<double> h_out(out_len);
    if (!read_vector(q_path, h_q) || !read_vector(k_path, h_k) || !read_vector(v_path, h_v)) return 5;
    if (p.has_mask && (strcmp(mask_path, "null") == 0 || !read_vector(mask_path, h_mask))) return 6;

    double *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_mask = NULL, *d_out = NULL;
    cudaMalloc((void**)&d_q, q_len * sizeof(double));
    cudaMalloc((void**)&d_k, k_len * sizeof(double));
    cudaMalloc((void**)&d_v, v_len * sizeof(double));
    cudaMalloc((void**)&d_out, out_len * sizeof(double));
    cudaMemcpy(d_q, h_q.data(), q_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_len * sizeof(double), cudaMemcpyHostToDevice);
    if (p.has_mask) {
        cudaMalloc((void**)&d_mask, mask_len * sizeof(double));
        cudaMemcpy(d_mask, h_mask.data(), mask_len * sizeof(double), cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    attention_kernel<<<blocks, threads>>>(d_q, d_k, d_v, d_mask, d_out, p, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(double), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) return 7;
    fwrite(h_out.data(), sizeof(double), out_len, fp);
    fclose(fp);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    if (d_mask) cudaFree(d_mask);
    return 0;
}
