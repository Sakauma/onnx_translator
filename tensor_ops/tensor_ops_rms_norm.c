/**
  ******************************************************************************
  * @file        tensor_ops_rms_norm.c
  * @author      Egor Izmaylov
  * @brief       实现 RMSNormalization C 后端算子。
  * @details     2026.06.28  V1.0.0  从 normalization shard 拆分 RMSNormalization。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 根据输出元素坐标计算 scale 在单向广播规则下对应的元素索引。
static size_t rms_scale_broadcast_index(const Tensor* input, const Tensor* scale, size_t output_index) {
    if (!input || !scale || !scale->data || scale->size == 0) return (size_t)-1;
    if (scale->ndim == 0 || scale->size == 1) return 0;
    if (scale->ndim > input->ndim || input->ndim > MAX_NDIM || scale->ndim > MAX_NDIM) return (size_t)-1;

    int input_coords[MAX_NDIM] = {0};
    int scale_coords[MAX_NDIM] = {0};
    get_coords_from_index(output_index, input_coords, input->shape, input->ndim);

    int offset = input->ndim - scale->ndim;
    for (int i = 0; i < scale->ndim; i++) {
        int input_axis = offset + i;
        int scale_dim = scale->shape[i];
        if (scale_dim == 1) {
            scale_coords[i] = 0;
        } else if (scale_dim == input->shape[input_axis]) {
            scale_coords[i] = input_coords[input_axis];
        } else {
            return (size_t)-1;
        }
    }

    return get_index_from_coords(scale_coords, scale->shape, scale->ndim);
}


// 实现 `rms normalization` 算子的 C 后端入口，按 axis 后缀计算 RMS 并应用 scale 广播。
void rms_normalization_forward(const Tensor* input, const Tensor* scale, Tensor* output,
                               int axis, float epsilon, int stash_type) {
    if (!input || !scale || !output || input->size != output->size) return;
    if (input->ndim <= 0 || input->ndim > MAX_NDIM) return;

    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;
    if (scale->ndim > ndim || scale->ndim > MAX_NDIM) return;

    size_t normalized_size = 1;
    for (int i = axis; i < ndim; i++) normalized_size *= (size_t)input->shape[i];
    if (normalized_size == 0) return;
    size_t row_count = input->size / normalized_size;
    int use_double_stash = (stash_type == 11);

    #pragma omp parallel for
    for (size_t row = 0; row < row_count; row++) {
        size_t row_offset = row * normalized_size;
        if (use_double_stash) {
            double square_sum = 0.0;
            for (size_t j = 0; j < normalized_size; j++) {
                double x = get_value_as_double(input, row_offset + j);
                square_sum += x * x;
            }
            double inv_rms = 1.0 / sqrt(square_sum / (double)normalized_size + (double)epsilon);
            for (size_t j = 0; j < normalized_size; j++) {
                size_t out_idx = row_offset + j;
                size_t scale_idx = rms_scale_broadcast_index(input, scale, out_idx);
                if (scale_idx == (size_t)-1) continue;
                double x = get_value_as_double(input, out_idx);
                double s = get_value_as_double(scale, scale_idx);
                set_tensor_value_from_float(output, out_idx, x * inv_rms * s);
            }
        } else {
            float square_sum = 0.0f;
            for (size_t j = 0; j < normalized_size; j++) {
                float x = (float)get_value_as_double(input, row_offset + j);
                square_sum += x * x;
            }
            float inv_rms = 1.0f / sqrtf(square_sum / (float)normalized_size + epsilon);
            for (size_t j = 0; j < normalized_size; j++) {
                size_t out_idx = row_offset + j;
                size_t scale_idx = rms_scale_broadcast_index(input, scale, out_idx);
                if (scale_idx == (size_t)-1) continue;
                float x = (float)get_value_as_double(input, out_idx);
                float s = (float)get_value_as_double(scale, scale_idx);
                set_tensor_value_from_float(output, out_idx, (double)(x * inv_rms * s));
            }
        }
    }
}
