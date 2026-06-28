/**
  ******************************************************************************
  * @file        tensor_ops_reduce_logsumexp.c
  * @author      Egor Izmaylov
  * @brief       实现稳定版 ReduceLogSumExp C 后端算子。
  * @details     2026.06.28  V1.0.0  从 reduce/arg shard 拆分 ReduceLogSumExp。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// ReduceLogSumExp 使用官方 reference 的稳定形式：先减去归约窗口最大值，再执行 exp/sum/log，避免大输入溢出成 Inf。
static void reduce_log_sum_exp_stable_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        double max_value = -INFINITY;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            double val = get_value_as_double(input, in_idx);
            double candidate = isinf(val) ? -INFINITY : val;
            if (candidate > max_value) {
                max_value = candidate;
            }
        }

        double sum = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            double val = get_value_as_double(input, in_idx);
            sum += exp(val - max_value);
        }

        set_tensor_value_from_float(output, i, log(sum) + max_value);
    }
}


// ReduceLogSumExp: Log(Sum(exp(x)))，仅实现基础定义
void reduce_log_sum_exp_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_log_sum_exp_stable_forward(input, output, params);
}
