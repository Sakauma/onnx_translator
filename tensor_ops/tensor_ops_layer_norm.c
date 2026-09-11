/**
  ******************************************************************************
  * @file        tensor_ops_layer_norm.c
  * @author      Egor Izmaylov
  * @brief       实现 LayerNormalization FLOAT stash 精度路径。
  * @details     2026.09.11  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// stash_type=FLOAT requires every stage-one operation to materialize in float32.
static void layer_norm_float_stash_impl(const Tensor* input, const Tensor* scale, const Tensor* B,
                                        Tensor* output, Tensor* mean_output, Tensor* inv_std_output,
                                        int axis, float epsilon) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;
    size_t norm_dim = 1;
    for (int i = axis; i < ndim; i++) norm_dim *= (size_t)input->shape[i];
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= (size_t)input->shape[i];

    #pragma omp parallel for
    for (size_t row = 0; row < outer_size; row++) {
        size_t offset = row * norm_dim;
        float sum = 0.0f;
        for (size_t col = 0; col < norm_dim; col++) sum = sum + get_value_as_float(input, offset + col);
        float mean = sum / (float)norm_dim;
        float square_sum = 0.0f;
        for (size_t col = 0; col < norm_dim; col++) {
            float diff = get_value_as_float(input, offset + col) - mean;
            float square = diff * diff;
            square_sum = square_sum + square;
        }
        float variance = square_sum / (float)norm_dim;
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        if (mean_output) set_tensor_value_from_float(mean_output, row, mean);
        if (inv_std_output) set_tensor_value_from_float(inv_std_output, row, inv_std);
        for (size_t col = 0; col < norm_dim; col++) {
            float diff = get_value_as_float(input, offset + col) - mean;
            float normalized = diff * inv_std;
            if (input->dtype == DTYPE_FLOAT32) {
                float y = normalized;
                if (scale) y = y * get_value_as_float(scale, col);
                if (B) y = y + get_value_as_float(B, col);
                set_tensor_value_from_float(output, offset + col, y);
            } else {
                double y = (double)normalized;
                if (scale) y *= get_value_as_double(scale, col);
                if (B) y += get_value_as_double(B, col);
                set_tensor_value_from_float(output, offset + col, y);
            }
        }
    }
}

void layer_norm_float_stash_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                    Tensor* output, int axis, float epsilon) {
    layer_norm_float_stash_impl(input, scale, B, output, NULL, NULL, axis, epsilon);
}

void layer_norm_float_stash_multi_output_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                                 Tensor* output, Tensor* mean_output, Tensor* inv_std_output,
                                                 int axis, float epsilon) {
    if (!mean_output || !inv_std_output) return;
    layer_norm_float_stash_impl(input, scale, B, output, mean_output, inv_std_output, axis, epsilon);
}
