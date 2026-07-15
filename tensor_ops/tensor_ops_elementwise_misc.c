/**
  ******************************************************************************
  * @file        tensor_ops_elementwise_misc.c
  * @author      Egor Izmaylov
  * @brief       实现逐元素杂项 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// Neg
// 实现 `neg` 算子的 C 后端入口，整数路径按目标 dtype 位宽回绕，匹配 ONNX reference 的 NumPy 行为。
void neg_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            int64_t val = get_value_as_int64(input, i);
            set_integer_value_wrapped(output, i, 0ULL - (uint64_t)val);
        } else {
            double val = get_value_as_double(input, i);
            set_tensor_value_from_float(output, i, -val);
        }
    }
}


// 实现 `sum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void sum_forward(const Tensor** inputs, int num_inputs, Tensor* output) {
    if (!inputs || !output || num_inputs < 1) return;
    for (int k = 0; k < num_inputs; k++) {
        if (!inputs[k] || inputs[k]->size != output->size) return;
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        double sum = 0.0;
        for (int k = 0; k < num_inputs; k++) {
            sum += get_value_as_double(inputs[k], i);
        }
        set_tensor_value_from_float(output, i, sum);
    }
}
