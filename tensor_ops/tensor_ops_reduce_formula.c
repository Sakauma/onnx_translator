/**
  ******************************************************************************
  * @file        tensor_ops_reduce_formula.c
  * @author      Egor Izmaylov
  * @brief       实现公式类归约 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

typedef enum {
    REDUCE_FORMULA_L1 = 0,
    REDUCE_FORMULA_L2 = 1,
    REDUCE_FORMULA_SUM_SQUARE = 2,
} ReduceFormulaMode;

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


// ReduceSumSquare: Sum(x^2)
void reduce_sum_square_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype)) {
        reduce_integer_formula_forward(input, output, params, REDUCE_FORMULA_SUM_SQUARE);
        return;
    }
    reduce_float_formula_forward(input, output, params, REDUCE_FORMULA_SUM_SQUARE);
}
