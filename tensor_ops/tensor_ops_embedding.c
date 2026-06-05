/**
  ******************************************************************************
  * @file        tensor_ops_embedding.c
  * @author      Egor Izmaylov
  * @brief       实现嵌入和位置编码相关的 C 后端算子。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 根据 ONNX RotaryEmbedding 的原始输入布局生成扁平索引。
static inline size_t rotary_embedding_index(int rank, int batch, int head, int sequence, int dim, int num_heads, int sequence_length, int head_size) {
    if (rank == 4) {
        return (((size_t)batch * (size_t)num_heads + (size_t)head) * (size_t)sequence_length + (size_t)sequence) * (size_t)head_size + (size_t)dim;
    }
    return ((size_t)batch * (size_t)sequence_length + (size_t)sequence) * ((size_t)num_heads * (size_t)head_size)
         + (size_t)head * (size_t)head_size + (size_t)dim;
}


// 根据 position_ids 或逐 token cache 布局读取当前 token 的 cos/sin 角度。
static inline size_t rotary_embedding_cache_index(const Tensor* cache, const Tensor* position_ids, int batch, int sequence, int pair, int sequence_length, int rotary_half) {
    if (position_ids && position_ids->data) {
        int64_t pos = get_value_as_int64(position_ids, (size_t)batch * (size_t)sequence_length + (size_t)sequence);
        if (pos < 0 || cache->ndim != 2 || pos >= cache->shape[0]) {
            return (size_t)-1;
        }
        return (size_t)pos * (size_t)rotary_half + (size_t)pair;
    }
    if (cache->ndim != 3 || batch >= cache->shape[0] || sequence >= cache->shape[1]) {
        return (size_t)-1;
    }
    return ((size_t)batch * (size_t)sequence_length + (size_t)sequence) * (size_t)rotary_half + (size_t)pair;
}


// 实现 `RotaryEmbedding` 的 C 后端入口，支持 3D/4D、position_ids、interleaved 和 partial rotation。
void rotary_embedding_forward(const Tensor* input, const Tensor* cos_cache, const Tensor* sin_cache,
                              const Tensor* position_ids, Tensor* output,
                              int num_heads, int rotary_embedding_dim, int interleaved) {
    if (!input || !cos_cache || !sin_cache || !output) return;
    if (!input->data || !cos_cache->data || !sin_cache->data || !output->data) return;
    if (input->size != output->size || input->ndim != output->ndim) return;
    if (input->ndim != 3 && input->ndim != 4) return;

    int rank = input->ndim;
    int batch_size = input->shape[0];
    int sequence_length = 0;
    int head_size = 0;
    if (rank == 4) {
        num_heads = input->shape[1];
        sequence_length = input->shape[2];
        head_size = input->shape[3];
    } else {
        sequence_length = input->shape[1];
        int hidden_size = input->shape[2];
        if (num_heads <= 0 || hidden_size % num_heads != 0) return;
        head_size = hidden_size / num_heads;
    }

    int rotary_dim = rotary_embedding_dim > 0 ? rotary_embedding_dim : head_size;
    if (batch_size <= 0 || sequence_length <= 0 || num_heads <= 0 || head_size <= 0) return;
    if (rotary_dim <= 0 || rotary_dim > head_size || (rotary_dim % 2) != 0) return;
    int rotary_half = rotary_dim / 2;
    if (cos_cache->ndim < 2 || sin_cache->ndim < 2) return;
    if (cos_cache->shape[cos_cache->ndim - 1] != rotary_half || sin_cache->shape[sin_cache->ndim - 1] != rotary_half) return;

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < sequence_length; ++s) {
                for (int d = 0; d < head_size; ++d) {
                    size_t out_index = rotary_embedding_index(rank, b, h, s, d, num_heads, sequence_length, head_size);
                    if (d >= rotary_dim) {
                        copy_tensor_element(output, out_index, input, out_index);
                        continue;
                    }

                    int pair = interleaved ? d / 2 : (d < rotary_half ? d : d - rotary_half);
                    size_t cache_index = rotary_embedding_cache_index(cos_cache, position_ids, b, s, pair, sequence_length, rotary_half);
                    if (cache_index == (size_t)-1) continue;

                    int real_dim = interleaved ? pair * 2 : pair;
                    int imag_dim = interleaved ? pair * 2 + 1 : pair + rotary_half;
                    size_t x1_index = rotary_embedding_index(rank, b, h, s, real_dim, num_heads, sequence_length, head_size);
                    size_t x2_index = rotary_embedding_index(rank, b, h, s, imag_dim, num_heads, sequence_length, head_size);
                    double x1 = get_value_as_double(input, x1_index);
                    double x2 = get_value_as_double(input, x2_index);
                    double cos_value = get_value_as_double(cos_cache, cache_index);
                    double sin_value = get_value_as_double(sin_cache, cache_index);
                    double rotated = (interleaved ? (d % 2 == 0) : (d < rotary_half))
                        ? cos_value * x1 - sin_value * x2
                        : sin_value * x1 + cos_value * x2;
                    set_tensor_value_from_float(output, out_index, rotated);
                }
            }
        }
    }
}
