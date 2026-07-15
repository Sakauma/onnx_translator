/**
  ******************************************************************************
  * @file        loss.h
  * @author      Egor Izmaylov
  * @brief       提供分类损失算子的空间展开和类别权重读取辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_loss.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_LOSS_H
#define TENSOR_OPS_INTERNAL_LOSS_H

#include "../tensor_ops_internal.h"

/* Loss 输入按 [N, C, spatial...] 解释；spatial_size 不包含 batch 和 class 维。 */

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
