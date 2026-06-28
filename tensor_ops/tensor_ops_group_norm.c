/**
  ******************************************************************************
  * @file        tensor_ops_group_norm.c
  * @author      Egor Izmaylov
  * @brief       实现 GroupNormalization C 后端算子。
  * @details     2026.06.28  V1.0.0  从 normalization shard 拆分 GroupNormalization。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// GroupNormalization
// 实现 `group norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void group_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                        Tensor* output, int num_groups, float epsilon) {
    if (!input || !scale || !B || !output) return;

    int N = input->shape[0];
    int C = input->shape[1];

    // 检查能否整除
    if (C % num_groups != 0) return;
    int channels_per_group = C / num_groups;

    // 计算空间大小 (H * W * ...)
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];

    // 每个 Group 的元素数量
    size_t group_size = channels_per_group * spatial_size;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            // 计算当前 Group 的 Mean 和 Var
            // Group 的数据范围：从 channel_start 到 channel_end
            int c_start = g * channels_per_group;
            int c_end = c_start + channels_per_group;

            double sum = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    sum += get_value_as_double(input, offset + i);
                }
            }
            double mean = sum / group_size;

            double sum_sq_diff = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double val = get_value_as_double(input, offset + i);
                    double diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
            }
            double var = sum_sq_diff / group_size;
            double inv_std = 1.0 / sqrt(var + epsilon);

            // 应用归一化和仿射变换
            for (int c = c_start; c < c_end; c++) {
                double s_val = get_value_as_double(scale, c);
                double b_val = get_value_as_double(B, c);

                double A = inv_std * s_val;
                double K = b_val - mean * A;

                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double x = get_value_as_double(input, offset + i);
                    double y = x * A + K;
                    set_tensor_value_from_float(output, offset + i, y);
                }
            }
        }
    }
}
