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

typedef enum {
    REDUCE_FORMULA_L1 = 0,
    REDUCE_FORMULA_L2 = 1,
    REDUCE_FORMULA_SUM_SQUARE = 2,
} ReduceFormulaMode;

// 初始化归约坐标，将输出坐标映射回输入坐标，归约轴先置零。
static inline void prepare_reduce_coords(
    size_t out_index,
    const Tensor* input,
    const Tensor* output,
    const ReduceParams* params,
    int* coords
) {
    int out_coords[MAX_NDIM];
    get_coords_from_index(out_index, out_coords, output->shape, output->ndim);

    if (params->keepdims) {
        for (int d = 0; d < input->ndim; d++) {
            coords[d] = is_axis_reduced(d, params->axes, params->num_axes) ? 0 : out_coords[d];
        }
        return;
    }

    int out_dim_idx = 0;
    for (int d = 0; d < input->ndim; d++) {
        if (is_axis_reduced(d, params->axes, params->num_axes)) {
            coords[d] = 0;
        } else {
            coords[d] = out_coords[out_dim_idx++];
        }
    }
}

// 计算归约轴组合数量，作为内层循环的展开空间。
static inline size_t reduce_total_steps_for(const Tensor* input, const ReduceParams* params) {
    size_t reduce_total_steps = 1;
    for (int i = 0; i < params->num_axes; i++) {
        reduce_total_steps *= input->shape[params->axes[i]];
    }
    return reduce_total_steps;
}

// 根据归约空间线性索引更新当前输入坐标。
static inline void update_reduce_coords(const Tensor* input, const ReduceParams* params, int* coords, size_t reduce_index) {
    size_t temp_r = reduce_index;
    for (int k = params->num_axes - 1; k >= 0; k--) {
        int axis_idx = params->axes[k];
        int dim_size = input->shape[axis_idx];
        coords[axis_idx] = temp_r % dim_size;
        temp_r /= dim_size;
    }
}

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

// 浮点公式归约保留实数公式语义，最后统一按输出 dtype 写回。
static void reduce_float_formula_forward(const Tensor* input, Tensor* output, ReduceParams* params, ReduceFormulaMode mode) {
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
            double val = get_value_as_double(input, in_idx);
            if (mode == REDUCE_FORMULA_L1) {
                acc += fabs(val);
            } else {
                acc += val * val;
            }
        }

        if (mode == REDUCE_FORMULA_L2) {
            acc = sqrt(acc);
        }
        set_tensor_value_from_float(output, i, acc);
    }
}

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

// 将非负浮点公式结果按 NumPy astype 的常见整数行为落到目标 dtype。
static inline uint64_t cast_formula_float_to_integer_raw(double value, DataType dtype) {
    int bits = integer_dtype_bits(dtype);
    if (is_unsigned_integer_dtype(dtype)) {
        if (isnan(value)) {
            return bits == 64 ? 0x8000000000000000ULL : 0ULL;
        }
        if (!isfinite(value) || value < 0.0) {
            return 0ULL;
        }
        long double limit = ldexpl(1.0L, bits);
        if ((long double)value >= limit) {
            return 0ULL;
        }
        return (uint64_t)value;
    }

    if (!isfinite(value) || isnan(value)) {
        return bits == 64 ? 0x8000000000000000ULL : (1ULL << (bits - 1));
    }
    long double max_value = ldexpl(1.0L, bits - 1) - 1.0L;
    long double min_value = -ldexpl(1.0L, bits - 1);
    if ((long double)value > max_value || (long double)value < min_value) {
        return bits == 64 ? 0x8000000000000000ULL : (1ULL << (bits - 1));
    }
    int64_t truncated = value < 0.0 ? (int64_t)ceil(value) : (int64_t)floor(value);
    return (uint64_t)truncated;
}

// 整数公式归约模拟 ONNX reference 中 NumPy 对整数 abs/square/sum/sqrt 的 dtype 行为。
static void reduce_integer_formula_forward(const Tensor* input, Tensor* output, ReduceParams* params, ReduceFormulaMode mode) {
    if (!input || !output || !params) return;
    size_t reduce_total_steps = reduce_total_steps_for(input, params);
    int bits = integer_dtype_bits(input->dtype);
    uint64_t mask = bits > 0 && bits < 64 ? ((1ULL << bits) - 1ULL) : UINT64_MAX;
    int unsigned_path = is_unsigned_integer_dtype(input->dtype);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
        prepare_reduce_coords(i, input, output, params, coords);

        uint64_t raw_acc = 0ULL;
        int64_t signed_acc = 0;
        int promoted_signed_acc = (!unsigned_path && bits < 64);
        int promoted_unsigned_acc = (unsigned_path && bits < 64);

        for (size_t r = 0; r < reduce_total_steps; r++) {
            update_reduce_coords(input, params, coords, r);
            size_t in_idx = get_index_from_coords(coords, input->shape, input->ndim);
            uint64_t raw_val = get_integer_value_as_uint64(input, in_idx);
            uint64_t term = raw_val;

            if (mode == REDUCE_FORMULA_L1) {
                if (!unsigned_path) {
                    int64_t signed_val = sign_extend_integer_bits(raw_val, bits);
                    term = signed_val < 0 ? (0ULL - raw_val) : raw_val;
                }
                term &= mask;
            } else {
                term = (raw_val * raw_val) & mask;
            }

            if (promoted_signed_acc) {
                signed_acc += sign_extend_integer_bits(term, bits);
            } else if (promoted_unsigned_acc) {
                raw_acc += term;
            } else {
                raw_acc = (raw_acc + term) & mask;
            }
        }

        if (mode == REDUCE_FORMULA_L2) {
            double sqrt_input;
            if (promoted_signed_acc) {
                sqrt_input = (double)signed_acc;
            } else if (promoted_unsigned_acc) {
                sqrt_input = (double)raw_acc;
            } else if (unsigned_path) {
                sqrt_input = (double)raw_acc;
            } else {
                sqrt_input = (double)sign_extend_integer_bits(raw_acc, bits);
            }
            double result = sqrt(sqrt_input);
            set_integer_value_wrapped(output, i, cast_formula_float_to_integer_raw(result, output->dtype));
        } else if (promoted_signed_acc) {
            set_integer_value_wrapped(output, i, (uint64_t)signed_acc);
        } else {
            set_integer_value_wrapped(output, i, raw_acc);
        }
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


// ReduceL1: Sum(|x|)
void reduce_l1_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_formula_forward(input, output, params, REDUCE_FORMULA_L1);
        return;
    }
    reduce_float_formula_forward(input, output, params, REDUCE_FORMULA_L1);
}


// ReduceL2: Sqrt(Sum(x^2))
void reduce_l2_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_formula_forward(input, output, params, REDUCE_FORMULA_L2);
        return;
    }
    reduce_float_formula_forward(input, output, params, REDUCE_FORMULA_L2);
}


// ReduceLogSumExp: Log(Sum(exp(x)))，仅实现基础定义
void reduce_log_sum_exp_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    reduce_log_sum_exp_stable_forward(input, output, params);
}


// ReduceSumSquare: Sum(x^2)
void reduce_sum_square_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_formula_forward(input, output, params, REDUCE_FORMULA_SUM_SQUARE);
        return;
    }
    reduce_float_formula_forward(input, output, params, REDUCE_FORMULA_SUM_SQUARE);
}
