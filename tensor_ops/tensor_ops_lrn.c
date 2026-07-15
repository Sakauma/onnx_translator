/**
  ******************************************************************************
  * @file        tensor_ops_lrn.c
  * @author      Egor Izmaylov
  * @brief       实现 LRN C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 实现 `lrn` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void lrn_forward(const Tensor* input, Tensor* output, int size, float alpha, float beta, float bias) {
    if (!input || !output || input->ndim < 3 || input->size != output->size || size <= 0) return;

    int channels = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    size_t batch_size = input->shape[0];
    int lower = (size - 1) / 2;
    int upper = size - 1 - lower;

    _Pragma("omp parallel for collapse(2)")
    for (size_t n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            int begin = c - lower;
            int end = c + upper + 1;
            if (begin < 0) begin = 0;
            if (end > channels) end = channels;

            for (size_t s = 0; s < spatial_size; s++) {
                double square_sum = 0.0;
                for (int cc = begin; cc < end; cc++) {
                    size_t idx = (n * (size_t)channels + (size_t)cc) * spatial_size + s;
                    double val = get_value_as_double(input, idx);
                    square_sum += val * val;
                }
                size_t out_idx = (n * (size_t)channels + (size_t)c) * spatial_size + s;
                double x = get_value_as_double(input, out_idx);
                double denom = pow((double)bias + ((double)alpha / (double)size) * square_sum, (double)beta);
                set_tensor_value_from_float(output, out_idx, x / denom);
            }
        }
    }
}
