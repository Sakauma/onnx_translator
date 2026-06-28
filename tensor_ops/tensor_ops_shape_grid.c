/**
  ******************************************************************************
  * @file        tensor_ops_shape_grid.c
  * @author      Egor Izmaylov
  * @brief       实现网格生成类 shape C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 将空间下标转换为 AffineGrid 使用的 [-1, 1] 规范化坐标。
static double affine_grid_normalized_coord(int index, int size, int align_corners) {
    if (size <= 1) {
        return 0.0;
    }
    if (align_corners) {
        return -1.0 + 2.0 * (double)index / (double)(size - 1);
    }
    return -1.0 + (2.0 * (double)index + 1.0) / (double)size;
}


// AffineGrid
// 实现 `affine grid` 算子的 C 后端入口，根据 theta 和 size 生成 2D/3D 规范化采样网格。
void affine_grid_forward(const Tensor* theta, const Tensor* size, Tensor* output, int align_corners) {
    if (!theta || !size || !output || !theta->data || !size->data || !output->data) return;
    if (theta->ndim != 3 || size->size < 4) return;

    int rank = (int)size->size - 2;
    if (rank != 2 && rank != 3) return;

    int batch = (int)get_value_as_int64(size, 0);
    if (batch <= 0) return;

    if (rank == 2) {
        int height = (int)get_value_as_int64(size, 2);
        int width = (int)get_value_as_int64(size, 3);
        if (height <= 0 || width <= 0) return;
        if (theta->shape[0] != batch || theta->shape[1] != 2 || theta->shape[2] != 3) return;
        if (output->ndim != 4 || output->shape[0] != batch || output->shape[1] != height ||
            output->shape[2] != width || output->shape[3] != 2) return;

        #pragma omp parallel for
        for (size_t i = 0; i < output->size; i++) {
            int coord = (int)(i % 2);
            int w = (int)((i / 2) % (size_t)width);
            int h = (int)((i / ((size_t)2 * width)) % (size_t)height);
            int n = (int)(i / ((size_t)2 * width * height));

            double x = affine_grid_normalized_coord(w, width, align_corners);
            double y = affine_grid_normalized_coord(h, height, align_corners);
            size_t theta_base = ((size_t)n * 2 * 3) + (size_t)coord * 3;
            double value = get_value_as_double(theta, theta_base) * x
                         + get_value_as_double(theta, theta_base + 1) * y
                         + get_value_as_double(theta, theta_base + 2);
            set_tensor_value_from_float(output, i, value);
        }
    } else {
        int depth = (int)get_value_as_int64(size, 2);
        int height = (int)get_value_as_int64(size, 3);
        int width = (int)get_value_as_int64(size, 4);
        if (depth <= 0 || height <= 0 || width <= 0) return;
        if (theta->shape[0] != batch || theta->shape[1] != 3 || theta->shape[2] != 4) return;
        if (output->ndim != 5 || output->shape[0] != batch || output->shape[1] != depth ||
            output->shape[2] != height || output->shape[3] != width || output->shape[4] != 3) return;

        #pragma omp parallel for
        for (size_t i = 0; i < output->size; i++) {
            int coord = (int)(i % 3);
            int w = (int)((i / 3) % (size_t)width);
            int h = (int)((i / ((size_t)3 * width)) % (size_t)height);
            int d = (int)((i / ((size_t)3 * width * height)) % (size_t)depth);
            int n = (int)(i / ((size_t)3 * width * height * depth));

            double x = affine_grid_normalized_coord(w, width, align_corners);
            double y = affine_grid_normalized_coord(h, height, align_corners);
            double z = affine_grid_normalized_coord(d, depth, align_corners);
            size_t theta_base = ((size_t)n * 3 * 4) + (size_t)coord * 4;
            double value = get_value_as_double(theta, theta_base) * x
                         + get_value_as_double(theta, theta_base + 1) * y
                         + get_value_as_double(theta, theta_base + 2) * z
                         + get_value_as_double(theta, theta_base + 3);
            set_tensor_value_from_float(output, i, value);
        }
    }
}
