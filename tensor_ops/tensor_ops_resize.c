/**
  ******************************************************************************
  * @file        tensor_ops_resize.c
  * @author      Egor Izmaylov
  * @brief       实现尺寸调整类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从 shape/index shard 拆分 Resize。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// Resize
// 实现 `resize` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void resize_forward(const Tensor* input, Tensor* output, float* scales, int coord_mode, int mode, int nearest_mode) {
    if (!input || !output || !scales) return;

    int ndim = input->ndim;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        get_coords_from_index(i, out_coords, output->shape, ndim);

        if (mode == 0) {
            // --- Nearest Neighbor ---
            int in_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;

                // 坐标变换
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f; // half_pixel
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f; // pytorch_half_pixel
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f; // align_corners
                else x_in = x_out / scale; // asymmetric (default)

                // 最近邻取整策略
                int in_idx = 0;
                if (nearest_mode == 2) {
                    // floor
                    in_idx = (int)floorf(x_in);
                } else if (nearest_mode == 3) {
                    // ceil
                    in_idx = (int)ceilf(x_in);
                } else {
                    // round_prefer_floor
                    in_idx = (int)ceilf(x_in - 0.5f);
                }
                // 边界截断 (Clamp)
                if (in_idx < 0) in_idx = 0;
                if (in_idx >= input->shape[d]) in_idx = input->shape[d] - 1;
                in_coords[d] = in_idx;
            }
            size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
            double val = get_value_as_double(input, in_idx);
            set_tensor_value_from_float(output, i, val);

        } else {
            // --- Linear Interpolation (N-Linear) ---
            // 计算每个维度的浮点坐标 x_in
            float real_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f;
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f;
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f;
                else x_in = x_out / scale;

                if (x_in < 0.0f) x_in = 0.0f;
                if (x_in > (float)(input->shape[d] - 1)) x_in = (float)(input->shape[d] - 1);

                real_coords[d] = x_in;
            }
            // N-Linear 插值核心
            int num_neighbors = 1 << ndim; // 2^ndim
            double weighted_sum = 0.0;
            for (int n = 0; n < num_neighbors; n++) {
                double weight = 1.0;
                int neighbor_coords[MAX_NDIM];
                for (int d = 0; d < ndim; d++) {
                    float x = real_coords[d];
                    int lower = (int)floorf(x);
                    int upper = lower + 1;
                    if (upper >= input->shape[d]) upper = input->shape[d] - 1;
                    // 检查当前邻居在维度 d 是取 Lower 还是 Upper
                    if ((n >> d) & 1) {
                        // 取 Upper
                        neighbor_coords[d] = upper;
                        weight *= (x - lower);
                    } else {
                        // 取 Lower
                        neighbor_coords[d] = lower;
                        weight *= (1.0f - (x - lower));
                    }
                }
                size_t n_idx = get_index_from_coords(neighbor_coords, input->shape, ndim);
                double val = get_value_as_double(input, n_idx);
                weighted_sum += val * weight;
            }
            set_tensor_value_from_float(output, i, weighted_sum);
        }
    }
}
