/**
  ******************************************************************************
  * @file        tensor_ops_reduce_arg_misc.c
  * @author      Egor Izmaylov
  * @brief       实现 Arg、ReduceLogSum 和 element-wise Mean C 后端算子。
  * @details     2026.06.28  V1.0.0  从 reduce/arg shard 拆分轻量入口。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// ArgMax 和 ArgMin
ARG_OP_IMPL(argmax_forward, -DBL_MAX, >, TENSOR_COMPARE_GT)


ARG_OP_IMPL(argmin_forward, DBL_MAX, <, TENSOR_COMPARE_LT)


// ReduceLogSum: Log(Sum(x))
REDUCE_OP_IMPL(reduce_log_sum_forward, 0.0, acc += val, acc = log(acc))


// Mean (Element-wise)
// 实现 `mean` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mean_forward(const Tensor** inputs, int num_inputs, Tensor* output) {
    if (!inputs || !output || num_inputs < 1) return;
    size_t size = output->size;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < size; i++) {
        double sum = 0.0;
        for (int k = 0; k < num_inputs; k++) {
            sum += get_value_as_double(inputs[k], i);
        }
        set_tensor_value_from_float(output, i, sum / num_inputs);
    }
}
