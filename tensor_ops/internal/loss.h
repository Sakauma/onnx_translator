/**
  ******************************************************************************
  * @file        loss.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_LOSS_H
#define TENSOR_OPS_INTERNAL_LOSS_H

#include "../tensor_ops_internal.h"

// 实现 `loss_spatial_size` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t loss_spatial_size(const Tensor* input) {
    size_t spatial = 1;
    for (int i = 2; i < input->ndim; i++) spatial *= (size_t)input->shape[i];
    return spatial;
}

// 实现 `loss_target_weight` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double loss_target_weight(const Tensor* weight, int64_t cls) {
    if (!weight) return 1.0;
    return get_value_as_double(weight, (size_t)cls);
}

#endif /* TENSOR_OPS_INTERNAL_LOSS_H */
