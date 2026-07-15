/**
  ******************************************************************************
  * @file        sort.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_SORT_H
#define TENSOR_OPS_INTERNAL_SORT_H

#include "../tensor_ops_internal.h"

// 用于排序
typedef struct {
    double value;
    uint64_t raw_value;
    int64_t signed_value;
    int64_t index;
} TopKElement;

#endif /* TENSOR_OPS_INTERNAL_SORT_H */
