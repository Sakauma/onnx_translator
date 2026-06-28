/**
  ******************************************************************************
  * @file        tensor_ops_compare_logic.c
  * @author      Egor Izmaylov
  * @brief       实现比较、逻辑、选择和标量符号类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


BINARY_COMP_IMPL(equal_forward, TENSOR_COMPARE_EQ)

BINARY_COMP_IMPL(greater_forward, TENSOR_COMPARE_GT)

BINARY_COMP_IMPL(less_forward, TENSOR_COMPARE_LT)

BINARY_COMP_IMPL(greater_or_equal_forward, TENSOR_COMPARE_GE)

BINARY_COMP_IMPL(less_or_equal_forward, TENSOR_COMPARE_LE)


// Not: 按位取反 (bool/uint8) 或 逻辑非
// 实现 `not` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void not_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        // ONNX Not 对 bool 生效，这里做逻辑非
        uint8_t res = (val == 0) ? 1 : 0; 
        set_tensor_value_from_int(output, i, res);
    }
}


// 实现 `isnan` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void isnan_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        uint8_t res = isnan(val) ? 1 : 0;
        set_tensor_value_from_int(output, i, res);
    }
}


BINARY_LOGIC_IMPL(and_forward, bool_a && bool_b)

BINARY_LOGIC_IMPL(or_forward,  bool_a || bool_b)

BINARY_LOGIC_IMPL(xor_forward, bool_a != bool_b)


UNARY_OP_IMPL(sin_forward, sin(val))

UNARY_OP_IMPL(tan_forward, tan(val))

UNARY_OP_IMPL(atan_forward, atan(val))


// 实现 `sign` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void sign_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res;
        if (isnan(val)) res = NAN;
        else if (val > 0) res = 1.0;
        else if (val < 0) res = -1.0;
        else res = 0.0;
        set_tensor_value_from_float(output, i, res);
    }
}


// 实现 `identity` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void identity_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
}


// 按 Python `%` 语义计算有符号整数余数，结果符号跟随除数。
static inline int64_t signed_python_mod(int64_t a, int64_t b) {
    if (b == 0 || b == -1) {
        return 0;
    }
    int64_t res = a % b;
    if (res != 0 && ((res < 0) != (b < 0))) {
        res += b;
    }
    return res;
}


// 实现 `mod` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mod_forward(const Tensor* A, const Tensor* B, Tensor* O, int fmod_mode) {
    if (!A || !B || !O) return;

    if (!fmod_mode && is_integer_dtype(A->dtype) && is_integer_dtype(B->dtype) && is_integer_dtype(O->dtype)) {
        int unsigned_path = O->dtype == DTYPE_UINT8 ||
                            O->dtype == DTYPE_UINT16 ||
                            O->dtype == DTYPE_UINT32 ||
                            O->dtype == DTYPE_UINT64;
        _Pragma("omp parallel for")
        for (size_t i = 0; i < O->size; i++) {
            if (unsigned_path) {
                uint64_t a = get_integer_value_as_uint64(A, i);
                uint64_t b = get_integer_value_as_uint64(B, i);
                uint64_t res = b == 0 ? 0 : a % b;
                set_integer_value_wrapped(O, i, res);
            } else {
                int64_t a = get_value_as_int64(A, i);
                int64_t b = get_value_as_int64(B, i);
                int64_t res = signed_python_mod(a, b);
                set_integer_value_wrapped(O, i, (uint64_t)res);
            }
        }
        return;
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double a = get_value_as_double(A, i);
        double b = get_value_as_double(B, i);
        double res;
        if (b == 0) {
            res = NAN;
        } else {
            if (fmod_mode) {
                res = fmod(a, b); 
            } else {
                res = a - floor(a / b) * b;
            }
        }
        set_tensor_value_from_float(O, i, res);
    }
}


// 实现 `where` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void where_forward(const Tensor* Cond, const Tensor* X, const Tensor* Y, Tensor* O) {
    if (!Cond || !X || !Y || !O) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double c_val = get_value_as_double(Cond, i);
        copy_tensor_element(O, i, (c_val != 0) ? X : Y, i);
    }
}
