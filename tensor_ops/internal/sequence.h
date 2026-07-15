/**
  ******************************************************************************
  * @file        sequence.h
  * @author      Egor Izmaylov
  * @brief       提供 Unique 稳定比较与 Mel 频率刻度转换辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_spectral_recurrent.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_SEQUENCE_H
#define TENSOR_OPS_INTERNAL_SEQUENCE_H

#include "../tensor_ops_internal.h"

/* 浮点排序将 NaN 放在有限值之后，并把两个 NaN 视为相等，以保持索引顺序稳定。 */

// 实现 `tensor_scalar_equal` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int tensor_scalar_equal(const Tensor* tensor, size_t lhs, size_t rhs) {
    if (!tensor || !tensor->data) return 0;
    if (IS_INT_TYPE(tensor->dtype)) {
        return get_value_as_int64(tensor, lhs) == get_value_as_int64(tensor, rhs);
    }
    double a = get_value_as_double(tensor, lhs);
    double b = get_value_as_double(tensor, rhs);
    if (isnan(a) && isnan(b)) return 1;
    return a == b;
}

// 作为 `tensor_scalar_compare` 排序比较函数，保证排序类算子的值和索引顺序稳定。
static int tensor_scalar_compare(const Tensor* tensor, size_t lhs, size_t rhs) {
    if (IS_INT_TYPE(tensor->dtype)) {
        int64_t a = get_value_as_int64(tensor, lhs);
        int64_t b = get_value_as_int64(tensor, rhs);
        return (a > b) - (a < b);
    }
    double a = get_value_as_double(tensor, lhs);
    double b = get_value_as_double(tensor, rhs);
    int a_nan = isnan(a);
    int b_nan = isnan(b);
    if (a_nan && b_nan) return 0;
    if (a_nan) return 1;
    if (b_nan) return -1;
    return (a > b) - (a < b);
}

// 实现 `unique` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `hz_to_mel` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double hz_to_mel(double frequency) {
    return 2595.0 * log10(1.0 + frequency / 700.0);
}

// 实现 `mel_to_hz` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

#endif /* TENSOR_OPS_INTERNAL_SEQUENCE_H */
