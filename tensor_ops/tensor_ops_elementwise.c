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
 * 初始化余弦查找表
 * 使用泰勒级数展开计算余弦值并存储在查找表中
 */
// 实现 `init_cos_lut` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
void init_cos_lut(void) {
    pthread_mutex_lock(&cos_lut_mutex);
    if (!cos_lut_initialized) {
        // 遍历查找表的每个位置
        for (int i = 0; i <= COS_LUT_SIZE; i++) {
            // 计算对应的角度值
            double x = (double)i * TWO_PI / COS_LUT_SIZE;
            double sign = 1.0;
            
            // 将角度映射到[0, π]区间
            if (x > PI) {
                x = TWO_PI - x;
            }
            // 将角度映射到[0, π/2]区间
            if (x > HALF_PI) {
                x = PI - x;
                sign = -1.0;
            }
            // 计算x的平方
            double x2 = x * x;
            double result;
            
            // 根据角度大小选择不同的计算方法
            if (x < 0.785398163397448) {
                // 使用余弦泰勒级数展开
                result = 1.0 + x2 * (-0.5 + x2 * (0.04166666666666666 +
                         x2 * (-0.001388888888888889 + x2 * 0.000024801587301587302)));
            } else {
                // 使用正弦泰勒级数展开，因为cos(x) = sin(π/2 - x)
                double t = HALF_PI - x;
                double t2 = t * t;
                result = t * (1.0 + t2 * (-0.16666666666666666 +
                         t2 * (0.008333333333333333 + t2 * (-0.0001984126984126984 +
                         t2 * 0.0000027557319223985893))));
            }
            // 存储带符号的计算结果
            cos_lut[i] = sign * result;
        }
        __sync_synchronize();
        // 标记查找表已初始化
        cos_lut_initialized = 1;
    }
    // 解锁
    pthread_mutex_unlock(&cos_lut_mutex);
}


/**
 * 余弦函数前向传播
 * * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `cos` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void cos_forward(const Tensor* input, Tensor* output) {
    // 检查输入参数是否有效
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }
    
    if (!cos_lut_initialized) init_cos_lut();

    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i); // 输入转 double
        double res = cos_lut_lookup(val);           // 查表
        set_tensor_value_from_float(output, i, res); // 安全写入输出
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


// 实现 `prelu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void prelu_forward(const Tensor* input, const Tensor* slope, Tensor* output) {
    if (!input || !slope || !output || input->size != output->size || slope->size != output->size) return;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        double x = get_value_as_double(input, i);
        double s = get_value_as_double(slope, i);
        double y = x >= 0.0 ? x : x * s;
        set_tensor_value_from_float(output, i, y);
    }
}


// Clip：支持全广播
// 调用此函数前，Python 端已将 input, min_t, max_t 广播为相同形状
// 实现 `clip` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void clip_forward(const Tensor* input, Tensor* output, const Tensor* min_t, const Tensor* max_t) {
    if (!input || !output) return;
    
    // 检查指针是否存在，避免空指针解引用
    int has_min = (min_t && min_t->data);
    int has_max = (max_t && max_t->data);

    if (is_integer_dtype(output->dtype) && is_integer_dtype(input->dtype)) {
        int unsigned_path = output->dtype == DTYPE_UINT8 ||
                            output->dtype == DTYPE_UINT16 ||
                            output->dtype == DTYPE_UINT32 ||
                            output->dtype == DTYPE_UINT64;
        #pragma omp parallel for
        for (size_t i = 0; i < output->size; i++) {
            if (unsigned_path) {
                uint64_t val = get_integer_value_as_uint64(input, i);
                if (has_min) {
                    uint64_t min_val = get_integer_value_as_uint64(min_t, i);
                    if (val < min_val) val = min_val;
                }
                if (has_max) {
                    uint64_t max_val = get_integer_value_as_uint64(max_t, i);
                    if (val > max_val) val = max_val;
                }
                set_integer_value_wrapped(output, i, val);
            } else {
                int64_t val = get_value_as_int64(input, i);
                if (has_min) {
                    int64_t min_val = get_value_as_int64(min_t, i);
                    if (val < min_val) val = min_val;
                }
                if (has_max) {
                    int64_t max_val = get_value_as_int64(max_t, i);
                    if (val > max_val) val = max_val;
                }
                set_integer_value_wrapped(output, i, (uint64_t)val);
            }
        }
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double val = get_value_as_double(input, i);
        if (has_min) {
            double min_val = get_value_as_double(min_t, i);
            if (val < min_val) val = min_val;
        }
        if (has_max) {
            double max_val = get_value_as_double(max_t, i);
            if (val > max_val) val = max_val;
        }
        set_tensor_value_from_float(output, i, val);
    }
}


// Elu: x > 0 ? x : alpha * (exp(x) - 1)
UNARY_OP_WITH_ALPHA_IMPL(elu_forward, (val > 0) ? val : a * (exp(val) - 1.0))


// LeakyRelu: x >= 0 ? x : alpha * x
UNARY_OP_WITH_ALPHA_IMPL(leaky_relu_forward, (val >= 0) ? val : a * val)


// ThresholdedRelu: x > alpha ? x : 0
UNARY_OP_WITH_ALPHA_IMPL(thresholded_relu_forward, (val > a) ? val : 0.0)


// Celu: x >= 0 ? x : alpha * (exp(x/alpha) - 1)
UNARY_OP_WITH_ALPHA_IMPL(celu_forward, (val >= 0) ? val : a * (exp(val / a) - 1.0))
