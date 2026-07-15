/**
  ******************************************************************************
  * @file        tensor_ops_elementwise.c
  * @author      Egor Izmaylov
  * @brief       实现逐元素二元算术类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"
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
