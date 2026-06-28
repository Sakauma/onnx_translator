/**
  ******************************************************************************
  * @file        tensor_ops_elementwise.c
  * @author      Egor Izmaylov
  * @brief       实现逐元素算术和基础激活类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


/**
 * ReLU激活函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `relu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void relu_forward(const Tensor* input, Tensor* output) {
    // 检查输入参数是否有效
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            // 整数路径 
            int64_t val = get_value_as_int64(input, i);
            int64_t res = val > 0 ? val : 0;
            set_tensor_value_from_int(output, i, res);
        } else {
            // 浮点路径
            double val = get_value_as_double(input, i);
            double res = val > 0 ? val : 0.0;
            set_tensor_value_from_float(output, i, res);
        }
    }
}


/**
 * Abs函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `abs` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void abs_forward(const Tensor* input, Tensor* output) {
    // 检查输入参数是否有效
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            // 整数路径
            int64_t val = get_value_as_int64(input, i);
            uint64_t res = val < 0 ? (0ULL - (uint64_t)val) : (uint64_t)val;
            set_integer_value_wrapped(output, i, res);
        } else {
            // 浮点路径
            double val = get_value_as_double(input, i);
            double res = fabs(val);
            set_tensor_value_from_float(output, i, res);
        }
    }
}


/**
 * Add函数前向传播实现
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `add` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void add_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    // 检查输入参数是否有效
    if (!A || !B || !O || !A->data || !B->data || !O->data || A->size != B->size || A->size != O->size) {
        return;
    }
    
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_add);
    } else {
        // 浮点路径
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) + get_value_as_double(B, i);
        } else {
            // 对所有非double浮点类型使用统一处理，包括float8
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                double val_a = get_value_as_double(A, i);
                double val_b = get_value_as_double(B, i);
                double res = val_a + val_b;
                set_tensor_value_from_float(O, i, res);
            }
        }
    }
}


/**
 * Sub函数前向传播实现 (A - B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `sub` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void sub_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    // 检查输入参数是否有效
    if (!A || !B || !O || !A->data || !B->data || !O->data || A->size != B->size || A->size != O->size) {
        return;
    }
    
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_sub);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) - get_value_as_double(B, i);
        } else {
            // 对所有非double浮点类型使用统一处理，包括float8
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                double val_a = get_value_as_double(A, i);
                double val_b = get_value_as_double(B, i);
                double res = val_a - val_b;
                set_tensor_value_from_float(O, i, res);
            }
        }
    }
}


/**
 * Mul函数前向传播实现 (A * B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `mul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mul_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    // 检查输入参数是否有效
    if (!A || !B || !O || !A->data || !B->data || !O->data || A->size != B->size || A->size != O->size) {
        return;
    }
    
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_mul);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) * get_value_as_double(B, i);
        } else {
            // 对所有非double浮点类型使用统一处理，包括float8
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                double val_a = get_value_as_double(A, i);
                double val_b = get_value_as_double(B, i);
                double res = val_a * val_b;
                set_tensor_value_from_float(O, i, res);
            }
        }
    }
}


/**
 * Div函数前向传播实现 (A / B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `div` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void div_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    // 检查输入参数是否有效
    if (!A || !B || !O || !A->data || !B->data || !O->data || A->size != B->size || A->size != O->size) {
        return;
    }
    
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_div);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                out_data[i] = get_value_as_double(A, i) / get_value_as_double(B, i);
            }
        } else {
            // 对所有非double浮点类型使用统一处理，包括float8
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                double val_a = get_value_as_double(A, i);
                double val_b = get_value_as_double(B, i);
                double res;
                res = val_a / val_b;
                set_tensor_value_from_float(O, i, res);
            }
        }
    }
}


// Exp 实现
UNARY_OP_IMPL(exp_forward, exp(val))


// Log 实现
// 未需要处理 log(0) 或负数的情况
UNARY_OP_IMPL(log_forward, log(val))


// Sqrt 实现
UNARY_OP_IMPL(sqrt_forward, sqrt(val))


// Sigmoid 实现
UNARY_OP_IMPL(sigmoid_forward, 1.0 / (1.0 + exp(-val)))


// Tanh 实现
UNARY_OP_IMPL(tanh_forward, tanh(val))


// Pow 实现
// 实现 `pow` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void pow_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    if (is_integer_dtype(A->dtype) && is_integer_dtype(B->dtype) && is_integer_dtype(O->dtype)) {
        _Pragma("omp parallel for")
        for (size_t i = 0; i < O->size; i++) {
            uint64_t base = get_integer_value_as_uint64(A, i);
            uint64_t exp = get_integer_value_as_uint64(B, i);
            uint64_t result = 1ULL;
            while (exp > 0) {
                if (exp & 1ULL) result *= base;
                exp >>= 1;
                if (exp) base *= base;
            }
            set_integer_value_wrapped(O, i, result);
        }
        return;
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double val_a = get_value_as_double(A, i);
        double val_b = get_value_as_double(B, i);
        double res = pow(val_a, val_b);
        set_tensor_value_from_float(O, i, res);
    }
}


// Max 实现
// 实现 `max` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void max_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;

    if (IS_INT_TYPE(O->dtype)) {
        // 整数路径
        BINARY_OP_INT_LOGIC(op_max);
    } else {
        // 浮点路径
        #pragma omp parallel for
        for (size_t i = 0; i < O->size; i++) {
            double val_a = get_value_as_double(A, i);
            double val_b = get_value_as_double(B, i);
            double res = (isnan(val_a) || isnan(val_b)) ? NAN : (val_a > val_b ? val_a : val_b);
            set_tensor_value_from_float(O, i, res);
        }
    }
}


// Min 实现
// 实现 `min` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void min_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;

    if (IS_INT_TYPE(O->dtype)) {
        // 整数路径：
        BINARY_OP_INT_LOGIC(op_min);
    } else {
        // 浮点路径
        #pragma omp parallel for
        for (size_t i = 0; i < O->size; i++) {
            double val_a = get_value_as_double(A, i);
            double val_b = get_value_as_double(B, i);
            double res = (isnan(val_a) || isnan(val_b)) ? NAN : (val_a < val_b ? val_a : val_b);
            set_tensor_value_from_float(O, i, res);
        }
    }
}


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


// Reciprocal
UNARY_OP_IMPL(reciprocal_forward, 1.0 / val)


// Ceil
UNARY_OP_IMPL(ceil_forward, ceil(val))


// Floor
UNARY_OP_IMPL(floor_forward, floor(val))


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
