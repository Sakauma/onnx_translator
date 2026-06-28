/**
  ******************************************************************************
  * @file        tensor_ops_activation_extra.c
  * @author      Egor Izmaylov
  * @brief       实现扩展激活、bitwise 和额外 unary math 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// Selu: gamma * (x > 0 ? x : alpha * (exp(x) - 1))
// 实现 `selu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void selu_forward(const Tensor* input, Tensor* output, float alpha, float gamma) {
    if (!input || !output) return;
    double a = (double)alpha;
    double g = (double)gamma;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res = g * ((val > 0) ? val : a * (exp(val) - 1.0));
        set_tensor_value_from_float(output, i, res);
    }
}


// HardSigmoid: max(0, min(1, alpha * x + beta))
// 实现 `hard sigmoid` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hard_sigmoid_forward(const Tensor* input, Tensor* output, float alpha, float beta) {
    if (!input || !output) return;
    double a = (double)alpha;
    double b = (double)beta;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res = fmax(0.0, fmin(1.0, a * val + b));
        set_tensor_value_from_float(output, i, res);
    }
}


// Softplus: ln(1 + exp(x))
UNARY_OP_IMPL(softplus_forward, log(1.0 + exp(val)))


// Softsign: x / (1 + |x|)
UNARY_OP_IMPL(softsign_forward, val / (1.0 + fabs(val)))


// HardSwish: x * max(0, min(1, alpha * x + beta)), default alpha=1/6, beta=0.5
// x * max(0, min(1, x/6 + 0.5))
UNARY_OP_IMPL(hard_swish_forward, val * fmax(0.0, fmin(1.0, val / 6.0 + 0.5)))


// 实现 `swish` 算子的 C 后端入口，按 ONNX 公式 x * sigmoid(alpha * x) 写回目标 dtype。
void swish_forward(const Tensor* input, Tensor* output, float alpha) {
    if (!input || !output) return;
    double a = (double)alpha;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double z = a * val;
        double sigmoid = z >= 0.0 ? 1.0 / (1.0 + exp(-z)) : exp(z) / (1.0 + exp(z));
        set_tensor_value_from_float(output, i, val * sigmoid);
    }
}


// Shrink: x < -lambd ? x + bias : (x > lambd ? x - bias : 0)
// 实现 `shrink` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void shrink_forward(const Tensor* input, Tensor* output, float bias, float lambd) {
    if (!input || !output) return;
    double b = (double)bias;
    double l = (double)lambd;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res;
        if (val < -l) res = val + b;
        else if (val > l) res = val - b;
        else res = 0.0;
        set_tensor_value_from_float(output, i, res);
    }
}


// Acos: arccos(x)
UNARY_OP_IMPL(acos_forward, acos(val))


// Asin: arcsin(x)
UNARY_OP_IMPL(asin_forward, asin(val))


// Cosh: (exp(x) + exp(-x)) / 2
UNARY_OP_IMPL(cosh_forward, cosh(val))


// Sinh: (exp(x) - exp(-x)) / 2
UNARY_OP_IMPL(sinh_forward, sinh(val))


// Asinh: ln(x + sqrt(x^2 + 1))
UNARY_OP_IMPL(asinh_forward, asinh(val))


// Acosh: ln(x + sqrt(x^2 - 1)), for x >= 1
UNARY_OP_IMPL(acosh_forward, acosh(val))


// Atanh: 0.5 * ln((1+x)/(1-x)), for |x| < 1
UNARY_OP_IMPL(atanh_forward, atanh(val))


// BitwiseAnd
// 实现 `bitwise and` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bitwise_and_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_and); 
}


// BitwiseOr
// 实现 `bitwise or` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bitwise_or_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_or);
}


// BitwiseXor
// 实现 `bitwise xor` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bitwise_xor_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_xor);
}


// BitwiseNot
// 实现 `bitwise not` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bitwise_not_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        uint64_t val = get_integer_value_as_uint64(input, i);
        uint64_t res = ~val;
        if (IS_INT_TYPE(output->dtype)) {
            set_integer_value_wrapped(output, i, res);
        } else {
            set_tensor_value_from_int(output, i, (int64_t)res);
        }
    }
}


// BitShift
// direction: 0=LEFT, 1=RIGHT
// 实现 `bit shift` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bit_shift_forward(const Tensor* A, const Tensor* B, Tensor* O, int direction) {
    if (!A || !B || !O) return;
    
    if (direction == 0) {
        // Left Shift
        BINARY_OP_INT_LOGIC(op_shift_left);
    } else {
        // Right Shift
        BINARY_OP_INT_LOGIC(op_shift_right);
    }
}


// IsInf
// 实现 `isinf` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void isinf_forward(const Tensor* input, Tensor* output, int detect_pos, int detect_neg) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        int res = 0;
        if (isinf(val)) {
            if (val > 0 && detect_pos) res = 1;
            else if (val < 0 && detect_neg) res = 1;
        }
        ((uint8_t*)output->data)[i] = (uint8_t)res;
    }
}


// ================== Group 7: Normalization & Math Extensions 实现 ==================

// Round: round to nearest integer
UNARY_OP_IMPL(round_forward, rint(val))


// Erf: error function
UNARY_OP_IMPL(erf_forward, erf(val))


// Gelu
static inline double gelu_exact_value(double val) {
    return 0.5 * val * (1.0 + erf(val * M_SQRT1_2));
}

static inline double gelu_tanh_value(double val) {
    const double sqrt_2_over_pi = 0.7978845608028654;
    return 0.5 * val * (1.0 + tanh(sqrt_2_over_pi * (val + 0.044715 * val * val * val)));
}

UNARY_OP_IMPL(gelu_forward, gelu_exact_value(val))

// 实现 `gelu` 算子的近似模式入口，0 使用精确 erf 公式，1 使用 ONNX tanh 近似公式。
void gelu_forward_mode(const Tensor* input, Tensor* output, int approximate_mode) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res = approximate_mode == 1 ? gelu_tanh_value(val) : gelu_exact_value(val);
        set_tensor_value_from_float(output, i, res);
    }
}


// 实现 `mish` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mish_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double sp;
        if (val > 20.0) sp = val;
        else sp = log(1.0 + exp(val));
        
        double res = val * tanh(sp);
        set_tensor_value_from_float(output, i, res);
    }
}


// Binarizer
// 实现 `binarizer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void binarizer_forward(const Tensor* input, Tensor* output, float threshold) {
    if (!input || !output) return;
    double t = (double)threshold;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        double res = (val > t) ? 1.0 : 0.0;
        set_tensor_value_from_float(output, i, res);
    }
}
