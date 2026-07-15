/**
  ******************************************************************************
  * @file        sort.h
  * @author      Egor Izmaylov
  * @brief       定义 TopK 排序时同时保留数值、原始整数位和索引的内部记录。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_sort_scan.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_SORT_H
#define TENSOR_OPS_INTERNAL_SORT_H

#include "../tensor_ops_internal.h"

/* raw_value/signed_value 避免整数 TopK 先转 double 后丢失 64 位精度。 */

// 用于排序
typedef struct {
    double value;
    uint64_t raw_value;
    int64_t signed_value;
    int64_t index;
} TopKElement;

#endif /* TENSOR_OPS_INTERNAL_SORT_H */
