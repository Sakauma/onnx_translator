/**
  ******************************************************************************
  * @file        tensor_ops_unary_basic.c
  * @author      Egor Izmaylov
  * @brief       实现基础一元数学和激活类 C 后端算子。
  * @details     2026.07.15  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 实现 `relu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void relu_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            int64_t val = get_value_as_int64(input, i);
            int64_t res = val > 0 ? val : 0;
            set_tensor_value_from_int(output, i, res);
        } else {
            double val = get_value_as_double(input, i);
            double res = val > 0 ? val : 0.0;
            set_tensor_value_from_float(output, i, res);
        }
    }
}

// 实现 `abs` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void abs_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            int64_t val = get_value_as_int64(input, i);
            uint64_t res = val < 0 ? (0ULL - (uint64_t)val) : (uint64_t)val;
            set_integer_value_wrapped(output, i, res);
        } else {
            double val = get_value_as_double(input, i);
            double res = fabs(val);
            set_tensor_value_from_float(output, i, res);
        }
    }
}

UNARY_OP_IMPL(exp_forward, exp(val))
UNARY_OP_IMPL(log_forward, log(val))
UNARY_OP_IMPL(sqrt_forward, sqrt(val))
UNARY_OP_IMPL(sigmoid_forward, 1.0 / (1.0 + exp(-val)))
UNARY_OP_IMPL(tanh_forward, tanh(val))
UNARY_OP_IMPL(reciprocal_forward, 1.0 / val)
UNARY_OP_IMPL(ceil_forward, ceil(val))
UNARY_OP_IMPL(floor_forward, floor(val))
