/**
  ******************************************************************************
  * @file        tensor_ops_attention.c
  * @author      Egor Izmaylov
  * @brief       实现 Attention 算子的 C 后端主数值路径。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 根据广播规则读取 Attention mask；当 mask 最后一维短于 KV 长度时返回 padded 标记。
static int attention_mask_index(const Tensor* mask, int batch, int head, int query, int key, size_t* index) {
    if (!mask || !mask->data || !index || mask->ndim <= 0 || mask->ndim > 4) return 0;
    int target_coords[4] = {batch, head, query, key};
    int offset = 4 - mask->ndim;
    size_t flat = 0;
    size_t stride = 1;
    for (int dim = mask->ndim - 1; dim >= 0; --dim) {
        int coord = target_coords[dim + offset];
        int size = mask->shape[dim];
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


// 按官方规则把 mask、causal 和 softcap 应用到单个 QK 分数上。
static double attention_score_with_bias(const Tensor* Q, const Tensor* K, const Tensor* attn_mask,
                                        int batch, int q_head, int kv_head,
                                        int query, int key, int head_size,
                                        int q_seq, int kv_seq,
                                        double scale, int is_causal, double softcap) {
    double score = 0.0;
    for (int dim = 0; dim < head_size; ++dim) {
        size_t q_idx = (((size_t)batch * (size_t)Q->shape[1] + (size_t)q_head) * (size_t)q_seq + (size_t)query)
                     * (size_t)head_size + (size_t)dim;
        size_t k_idx = (((size_t)batch * (size_t)K->shape[1] + (size_t)kv_head) * (size_t)kv_seq + (size_t)key)
                     * (size_t)head_size + (size_t)dim;
        score += get_value_as_double(Q, q_idx) * get_value_as_double(K, k_idx);
    }
    score *= scale;

    if (attn_mask && attn_mask->data) {
        size_t mask_idx = 0;
        if (attention_mask_index(attn_mask, batch, q_head, query, key, &mask_idx)) {
            if (attn_mask->dtype == DTYPE_BOOL) {
                if (get_value_as_double(attn_mask, mask_idx) == 0.0) {
                    score = -INFINITY;
                }
            } else {
                score += get_value_as_double(attn_mask, mask_idx);
            }
        } else {
            score = -INFINITY;
        }
    }

    if (is_causal && key > query) {
        score = -INFINITY;
    }
    if (softcap > 0.0) {
        score = tanh(score / softcap) * softcap;
    }
    return score;
}


// 执行 4D Attention C 后端入口，使用 double 累加并按输出 dtype 写回。
void attention_forward(const Tensor* Q, const Tensor* K, const Tensor* V,
                       const Tensor* attn_mask, Tensor* Y,
                       int q_num_heads, int kv_num_heads,
                       float scale, int is_causal, float softcap) {
    if (!Q || !K || !V || !Y) return;
    if (!Q->data || !K->data || !V->data || !Y->data) return;
    if (Q->ndim != 4 || K->ndim != 4 || V->ndim != 4 || Y->ndim != 4) return;

    int batch_size = Q->shape[0];
    int q_heads = Q->shape[1];
    int q_seq = Q->shape[2];
    int head_size = Q->shape[3];
    int kv_heads = K->shape[1];
    int kv_seq = K->shape[2];
    int v_head_size = V->shape[3];
    if (q_num_heads > 0 && q_num_heads != q_heads) return;
    if (kv_num_heads > 0 && kv_num_heads != kv_heads) return;
    if (K->shape[0] != batch_size || V->shape[0] != batch_size) return;
    if (K->shape[3] != head_size || V->shape[1] != kv_heads || V->shape[2] != kv_seq) return;
    if (Y->shape[0] != batch_size || Y->shape[1] != q_heads || Y->shape[2] != q_seq || Y->shape[3] != v_head_size) return;
    if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_size <= 0) return;

    double score_scale = scale >= 0.0f ? (double)scale : 1.0 / sqrt((double)head_size);
    double cap = softcap > 0.0f ? (double)softcap : 0.0;
    int head_repeat = q_heads / kv_heads;

    #pragma omp parallel for collapse(3)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int q_head = 0; q_head < q_heads; ++q_head) {
            for (int query = 0; query < q_seq; ++query) {
                int kv_head = q_head / head_repeat;
                double max_score = -INFINITY;
                for (int key = 0; key < kv_seq; ++key) {
                    double score = attention_score_with_bias(Q, K, attn_mask, batch, q_head, kv_head,
                                                             query, key, head_size, q_seq, kv_seq,
                                                             score_scale, is_causal, cap);
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                double denom = 0.0;
                for (int key = 0; key < kv_seq; ++key) {
                    double score = attention_score_with_bias(Q, K, attn_mask, batch, q_head, kv_head,
                                                             query, key, head_size, q_seq, kv_seq,
                                                             score_scale, is_causal, cap);
                    denom += exp(score - max_score);
                }

                for (int value_dim = 0; value_dim < v_head_size; ++value_dim) {
                    double sum = 0.0;
                    for (int key = 0; key < kv_seq; ++key) {
                        double score = attention_score_with_bias(Q, K, attn_mask, batch, q_head, kv_head,
                                                                 query, key, head_size, q_seq, kv_seq,
                                                                 score_scale, is_causal, cap);
                        double weight = exp(score - max_score) / denom;
                        size_t v_idx = (((size_t)batch * (size_t)kv_heads + (size_t)kv_head) * (size_t)kv_seq + (size_t)key)
                                     * (size_t)v_head_size + (size_t)value_dim;
                        sum += weight * get_value_as_double(V, v_idx);
                    }
                    size_t y_idx = (((size_t)batch * (size_t)q_heads + (size_t)q_head) * (size_t)q_seq + (size_t)query)
                                 * (size_t)v_head_size + (size_t)value_dim;
                    set_tensor_value_from_float(Y, y_idx, sum);
                }
            }
        }
    }
}
