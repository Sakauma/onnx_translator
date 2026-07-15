/**
  ******************************************************************************
  * @file        spectral_transform.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_SPECTRAL_TRANSFORM_H
#define TENSOR_OPS_INTERNAL_SPECTRAL_TRANSFORM_H

#include "../tensor_ops_internal.h"

// 实现 `complex_tensor_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t complex_tensor_index(const Tensor* tensor, const int* coords, int component) {
    int complex_rank = tensor->ndim - 1;
    size_t idx = 0;
    for (int d = 0; d < complex_rank; d++) {
        idx = idx * (size_t)tensor->shape[d] + (size_t)coords[d];
    }
    return idx * (size_t)tensor->shape[complex_rank] + (size_t)component;
}

// 封装 `get_complex_value` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static void get_complex_value(const Tensor* tensor, const int* coords, double* real, double* imag) {
    *real = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 0));
    *imag = 0.0;
    if (tensor->shape[tensor->ndim - 1] == 2) {
        *imag = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 1));
    }
}

// 实现 `normalize_complex_axis` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int normalize_complex_axis(int axis, int complex_rank) {
    if (axis < 0) axis += complex_rank + 1;
    return axis;
}

#endif /* TENSOR_OPS_INTERNAL_SPECTRAL_TRANSFORM_H */
