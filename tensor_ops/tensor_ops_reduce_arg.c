/*
 * 文件功能：实现归约、Arg 和均值类 C 后端算子。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include "tensor_ops_internal.h"


// ReduceSum: Init=0, Acc+=val
REDUCE_OP_IMPL(reduce_sum_forward, 0.0, acc += val, (void)0)

// ReduceMean: Init=0, Acc+=val, Post=acc/count
REDUCE_OP_IMPL(reduce_mean_forward, 0.0, acc += val, acc /= reduce_total_steps)

// ReduceProd: Init=1, Acc*=val
REDUCE_OP_IMPL(reduce_prod_forward, 1.0, acc *= val, (void)0)

// ReduceMax: Init=-inf, Acc=max
REDUCE_OP_IMPL(reduce_max_forward, -DBL_MAX, if(val > acc) acc = val, (void)0)

// ReduceMin: Init=+inf, Acc=min
REDUCE_OP_IMPL(reduce_min_forward, DBL_MAX, if(val < acc) acc = val, (void)0)


//ArgMax和ArgMin
ARG_OP_IMPL(argmax_forward, -DBL_MAX, >)


ARG_OP_IMPL(argmin_forward, DBL_MAX, <)


// ReduceL1: Sum(|x|)
REDUCE_OP_IMPL(reduce_l1_forward, 0.0, acc += fabs(val), (void)0)


// ReduceL2: Sqrt(Sum(x^2))
REDUCE_OP_IMPL(reduce_l2_forward, 0.0, acc += val * val, acc = sqrt(acc))


// ReduceLogSum: Log(Sum(x))
REDUCE_OP_IMPL(reduce_log_sum_forward, 0.0, acc += val, acc = log(acc))


// ReduceLogSumExp: Log(Sum(exp(x)))，仅实现基础定义
REDUCE_OP_IMPL(reduce_log_sum_exp_forward, 0.0, acc += exp(val), acc = log(acc))


// ReduceSumSquare: Sum(x^2)
REDUCE_OP_IMPL(reduce_sum_square_forward, 0.0, acc += val * val, (void)0)


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

