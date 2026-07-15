/**
  ******************************************************************************
  * @file        spatial.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_SPATIAL_H
#define TENSOR_OPS_INTERNAL_SPATIAL_H

#include "../tensor_ops_internal.h"

// 安全获取4D张量的值
// 封装 `get_val_4d_with_padding` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline double get_val_4d_with_padding(const Tensor* T, int n, int c, int h, int w, double pad_val) {
    int N = T->shape[0];
    int C = T->shape[1];
    int H = T->shape[2];
    int W = T->shape[3];

    // 越界检查：如果坐标在张量范围外，返回 padding 值
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) {
        return pad_val;
    }
    // 计算平坦索引
    size_t idx = ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)h * W) + w;
    return get_value_as_double(T, idx);
}

#endif /* TENSOR_OPS_INTERNAL_SPATIAL_H */
