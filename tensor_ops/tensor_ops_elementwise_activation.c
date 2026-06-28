/**
  ******************************************************************************
  * @file        tensor_ops_elementwise_activation.c
  * @author      Egor Izmaylov
  * @brief       实现带参数或范围约束的逐元素激活类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从基础 elementwise shard 拆分参数化激活。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `prelu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void prelu_forward(const Tensor* input, const Tensor* slope, Tensor* output) {
    if (!input || !slope || !output || input->size != output->size || slope->size != output->size) return;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        double x = get_value_as_double(input, i);
        double s = get_value_as_double(slope, i);
        double y = x >= 0.0 ? x : x * s;
        set_tensor_value_from_float(output, i, y);
    }
}


// Clip：支持全广播
// 调用此函数前，Python 端已将 input, min_t, max_t 广播为相同形状
// 实现 `clip` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void clip_forward(const Tensor* input, Tensor* output, const Tensor* min_t, const Tensor* max_t) {
    if (!input || !output) return;

    // 检查指针是否存在，避免空指针解引用
    int has_min = (min_t && min_t->data);
    int has_max = (max_t && max_t->data);

    if (is_integer_dtype(output->dtype) && is_integer_dtype(input->dtype)) {
        int unsigned_path = output->dtype == DTYPE_UINT8 ||
                            output->dtype == DTYPE_UINT16 ||
                            output->dtype == DTYPE_UINT32 ||
                            output->dtype == DTYPE_UINT64;
        #pragma omp parallel for
        for (size_t i = 0; i < output->size; i++) {
            if (unsigned_path) {
                uint64_t val = get_integer_value_as_uint64(input, i);
                if (has_min) {
                    uint64_t min_val = get_integer_value_as_uint64(min_t, i);
                    if (val < min_val) val = min_val;
                }
                if (has_max) {
                    uint64_t max_val = get_integer_value_as_uint64(max_t, i);
                    if (val > max_val) val = max_val;
                }
                set_integer_value_wrapped(output, i, val);
            } else {
                int64_t val = get_value_as_int64(input, i);
                if (has_min) {
                    int64_t min_val = get_value_as_int64(min_t, i);
                    if (val < min_val) val = min_val;
                }
                if (has_max) {
                    int64_t max_val = get_value_as_int64(max_t, i);
                    if (val > max_val) val = max_val;
                }
                set_integer_value_wrapped(output, i, (uint64_t)val);
            }
        }
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double val = get_value_as_double(input, i);
        if (has_min) {
            double min_val = get_value_as_double(min_t, i);
            if (val < min_val) val = min_val;
        }
        if (has_max) {
            double max_val = get_value_as_double(max_t, i);
            if (val > max_val) val = max_val;
        }
        set_tensor_value_from_float(output, i, val);
    }
}


// Elu: x > 0 ? x : alpha * (exp(x) - 1)
UNARY_OP_WITH_ALPHA_IMPL(elu_forward, (val > 0) ? val : a * (exp(val) - 1.0))


// LeakyRelu: x >= 0 ? x : alpha * x
UNARY_OP_WITH_ALPHA_IMPL(leaky_relu_forward, (val >= 0) ? val : a * val)


// ThresholdedRelu: x > alpha ? x : 0
UNARY_OP_WITH_ALPHA_IMPL(thresholded_relu_forward, (val > a) ? val : 0.0)


// Celu: x >= 0 ? x : alpha * (exp(x/alpha) - 1)
UNARY_OP_WITH_ALPHA_IMPL(celu_forward, (val >= 0) ? val : a * (exp(val / a) - 1.0))
