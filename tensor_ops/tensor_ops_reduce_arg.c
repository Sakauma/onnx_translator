/**
  ******************************************************************************
  * @file        tensor_ops_reduce_arg.c
  * @author      Egor Izmaylov
  * @brief       实现归约、Arg 和均值类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

typedef enum {
    REDUCE_NUMERIC_SUM = 0,
    REDUCE_NUMERIC_PROD = 1,
    REDUCE_NUMERIC_MAX = 2,
    REDUCE_NUMERIC_MIN = 3,
} ReduceNumericMode;

// 保留原有 double 累加/比较路径，服务浮点归约和低精度浮点写回。
static void reduce_float_numeric_forward(const Tensor* input, Tensor* output, ReduceParams* params, ReduceNumericMode mode) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        double acc = 0.0;
        if (mode == REDUCE_NUMERIC_PROD) acc = 1.0;
        else if (mode == REDUCE_NUMERIC_MAX) acc = -DBL_MAX;
        else if (mode == REDUCE_NUMERIC_MIN) acc = DBL_MAX;

        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            double val = get_value_as_double(input, in_idx);
            if (mode == REDUCE_NUMERIC_SUM) acc += val;
            else if (mode == REDUCE_NUMERIC_PROD) acc *= val;
            else if (mode == REDUCE_NUMERIC_MAX) {
                if (val > acc) acc = val;
            } else if (val < acc) {
                acc = val;
            }
        }

        set_tensor_value_from_float(output, i, acc);
    }
}

// 浮点 ReduceMean 先以 double 累加并完成除法，再按目标低精度 dtype 写回。
static void reduce_float_mean_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        double acc = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            acc += get_value_as_double(input, in_idx);
        }

        set_tensor_value_from_float(output, i, acc / (double)reduce_total_steps);
    }
}

// 整数归约按 dtype 位宽进行回绕，避免 int64/uint64 经过 double 后丢失低位。
static void reduce_integer_numeric_forward(const Tensor* input, Tensor* output, ReduceParams* params, ReduceNumericMode mode) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);
    int unsigned_path = is_unsigned_integer_dtype(input->dtype);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        uint64_t raw_acc = mode == REDUCE_NUMERIC_PROD ? 1ULL : 0ULL;
        int64_t signed_best = 0;
        int initialized = 0;

        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            uint64_t raw_val = get_integer_value_as_uint64(input, in_idx);

            if (mode == REDUCE_NUMERIC_SUM) {
                raw_acc += raw_val;
            } else if (mode == REDUCE_NUMERIC_PROD) {
                raw_acc *= raw_val;
            } else if (mode == REDUCE_NUMERIC_MAX || mode == REDUCE_NUMERIC_MIN) {
                if (!initialized) {
                    raw_acc = raw_val;
                    signed_best = get_value_as_int64(input, in_idx);
                    initialized = 1;
                } else if (unsigned_path) {
                    int update = mode == REDUCE_NUMERIC_MAX ? raw_val > raw_acc : raw_val < raw_acc;
                    if (update) raw_acc = raw_val;
                } else {
                    int64_t signed_val = get_value_as_int64(input, in_idx);
                    int update = mode == REDUCE_NUMERIC_MAX ? signed_val > signed_best : signed_val < signed_best;
                    if (update) {
                        raw_acc = raw_val;
                        signed_best = signed_val;
                    }
                }
            }
        }

        set_integer_value_wrapped(output, i, raw_acc);
    }
}

// ReduceMean 的整数 reference 语义等价于 np.mean(data, dtype=data.dtype)：先按 dtype 累加回绕，再除以元素数并向零截断。
static void reduce_integer_mean_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);
    int bits = integer_dtype_bits(input->dtype);
    uint64_t mask = bits > 0 && bits < 64 ? ((1ULL << bits) - 1ULL) : UINT64_MAX;
    int unsigned_path = is_unsigned_integer_dtype(input->dtype);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        uint64_t raw_sum = 0ULL;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            raw_sum += get_integer_value_as_uint64(input, in_idx);
            raw_sum &= mask;
        }

        double mean_value;
        if (unsigned_path) {
            mean_value = (double)raw_sum / (double)reduce_total_steps;
            if (!isfinite(mean_value) || mean_value < 0.0 || mean_value >= 18446744073709551616.0) {
                set_integer_value_wrapped(output, i, 0ULL);
            } else {
                set_integer_value_wrapped(output, i, (uint64_t)mean_value);
            }
        } else {
            int64_t signed_sum = sign_extend_integer_bits(raw_sum, bits);
            mean_value = (double)signed_sum / (double)reduce_total_steps;
            if (!isfinite(mean_value) || mean_value >= 9223372036854775808.0 || mean_value <= -9223372036854775808.0) {
                set_integer_value_wrapped(output, i, 0x8000000000000000ULL);
            } else if (mean_value < 0.0) {
                set_integer_value_wrapped(output, i, (uint64_t)(int64_t)ceil(mean_value));
            } else {
                set_integer_value_wrapped(output, i, (uint64_t)(int64_t)floor(mean_value));
            }
        }
    }
}

// 根据输入/输出 dtype 在整数精确路径和原有浮点路径之间调度。
static void reduce_numeric_forward(const Tensor* input, Tensor* output, ReduceParams* params, ReduceNumericMode mode) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_numeric_forward(input, output, params, mode);
        return;
    }
    reduce_float_numeric_forward(input, output, params, mode);
}


// ReduceSum: Init=0, Acc+=val
void reduce_sum_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_numeric_forward(input, output, params, REDUCE_NUMERIC_SUM);
}

// ReduceMean: Init=0, Acc+=val, Post=acc/count
void reduce_mean_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_mean_forward(input, output, params);
        return;
    }
    reduce_float_mean_forward(input, output, params);
}

// ReduceProd: Init=1, Acc*=val
void reduce_prod_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_numeric_forward(input, output, params, REDUCE_NUMERIC_PROD);
}

// ReduceMax: Init=-inf, Acc=max
void reduce_max_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_numeric_forward(input, output, params, REDUCE_NUMERIC_MAX);
}

// ReduceMin: Init=+inf, Acc=min
void reduce_min_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_numeric_forward(input, output, params, REDUCE_NUMERIC_MIN);
}
