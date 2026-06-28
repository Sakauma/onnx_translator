/**
  ******************************************************************************
  * @file        tensor_ops_matrix_quant.c
  * @author      Egor Izmaylov
  * @brief       实现矩阵乘、Gemm 和量化类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 根据 ONNX precision 属性选择除法精度；0 表示沿用既有默认路径。
static int quantize_linear_use_double_precision(const Tensor* X, const Tensor* Scale, int precision) {
    if (precision == 11) return 1;  // ONNX TensorProto.DOUBLE
    if (precision == 1 || precision == 10 || precision == 16) return 0;  // FLOAT/FLOAT16/BFLOAT16
    return X->dtype == DTYPE_FLOAT64 || Scale->dtype == DTYPE_FLOAT64;
}


// 按 QuantizeLinear 的 saturate 属性写入 float8 输出；整数输出仍由通用饱和写回处理。
static uint8_t quantize_float_to_fp8_e4m3(float f, int saturate) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255 && mant != 0) return (uint8_t)(0x7F | sign);
    if (exp == 255) return (uint8_t)((saturate ? 0x7E : 0x7F) | sign);
    if (exp == 0) return (uint8_t)sign;

    exp = exp - 127 + 7;
    if (exp < 1) return (uint8_t)sign;

    uint32_t mant_3 = mant >> 20;
    uint32_t rem = mant & 0xFFFFF;
    if (rem > 0x80000 || (rem == 0x80000 && (mant_3 & 1))) {
        mant_3 += 1;
        if (mant_3 > 7) {
            mant_3 = 0;
            exp += 1;
        }
    }

    if (exp > 15 || (exp == 15 && mant_3 == 7)) {
        return (uint8_t)((saturate ? 0x7E : 0x7F) | sign);
    }
    return (uint8_t)(sign | ((uint32_t)exp << 3) | mant_3);
}


// 按 QuantizeLinear 的 saturate 属性写入 float8 输出；E5M2 非饱和溢出保留 Inf。
static uint8_t quantize_float_to_fp8_e5m2(float f, int saturate) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255 && mant != 0) return (uint8_t)(sign | 0x7D);
    if (exp == 255) return (uint8_t)(sign | (saturate ? 0x7B : 0x7C));
    if (exp == 0) return (uint8_t)sign;

    exp = exp - 127 + 15;
    if (exp < 1) return (uint8_t)sign;

    uint32_t mant_2 = mant >> 21;
    uint32_t rem = mant & 0x1FFFFF;
    if (rem > 0x100000 || (rem == 0x100000 && (mant_2 & 1))) {
        mant_2 += 1;
        if (mant_2 > 3) {
            mant_2 = 0;
            exp += 1;
        }
    }
    if (exp >= 31) return (uint8_t)(sign | (saturate ? 0x7B : 0x7C));
    return (uint8_t)(sign | ((uint32_t)exp << 2) | mant_2);
}


// 将 QuantizeLinear 结果写入目标张量，float8 使用属性控制的专用溢出语义。
static void set_quantize_linear_value(Tensor* Y, size_t index, double value, int saturate) {
    if (Y->dtype == DTYPE_FLOAT8_E4M3) {
        ((uint8_t*)Y->data)[index] = quantize_float_to_fp8_e4m3((float)value, saturate);
    } else if (Y->dtype == DTYPE_FLOAT8_E5M2) {
        ((uint8_t*)Y->data)[index] = quantize_float_to_fp8_e5m2((float)value, saturate);
    } else if (Y->dtype == DTYPE_FLOAT8_E4M3FNUZ) {
        ((uint8_t*)Y->data)[index] = float_to_fp8_e4m3fnuz_saturate((float)value, saturate);
    } else if (Y->dtype == DTYPE_FLOAT8_E5M2FNUZ) {
        ((uint8_t*)Y->data)[index] = float_to_fp8_e5m2fnuz_saturate((float)value, saturate);
    } else if (Y->dtype == DTYPE_FLOAT4_E2M1) {
        ((uint8_t*)Y->data)[index] = float_to_fp4_e2m1((float)value);
    } else if (Y->dtype == DTYPE_FLOAT8_E8M0) {
        ((uint8_t*)Y->data)[index] = float_to_fp8_e8m0((float)value);
    } else {
        set_tensor_value_from_float(Y, index, value);
    }
}


// 判断 QuantizeLinear 输出是否为浮点量化格式；该类 dtype 直接舍入到目标浮点格式，不先执行整数 rint。
static int quantize_linear_output_is_float_dtype(DataType dtype) {
    return dtype == DTYPE_FLOAT8_E4M3 ||
           dtype == DTYPE_FLOAT8_E5M2 ||
           dtype == DTYPE_FLOAT8_E4M3FNUZ ||
           dtype == DTYPE_FLOAT8_E5M2FNUZ ||
           dtype == DTYPE_FLOAT4_E2M1 ||
           dtype == DTYPE_FLOAT8_E8M0;
}


// 实现 `quantize linear` 的共享计算逻辑，支持默认精度和显式 precision/saturate 属性。
static void quantize_linear_forward_impl(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y, int precision, int saturate) {
    if (!X || !Scale || !ZeroPoint || !Y) return;
    
    size_t loop_size = Y->size;
    int use_double_precision = quantize_linear_use_double_precision(X, Scale, precision);
    int output_is_float_dtype = quantize_linear_output_is_float_dtype(Y->dtype);

    #pragma omp parallel for
    for (size_t i = 0; i < loop_size; i++) {
        double zp_val = get_value_as_double(ZeroPoint, i);
        
        double res = zp_val; 
        if (use_double_precision) {
            double x_val = get_value_as_double(X, i);
            double s_val = get_value_as_double(Scale, i);
            if (s_val != 0.0) {
                double scaled = x_val / s_val + zp_val;
                res = output_is_float_dtype ? scaled : rint(x_val / s_val) + zp_val;
            }
        } else {
            float x_val = get_value_as_float(X, i);
            float s_val = get_value_as_float(Scale, i);
            float zp_float = (float)zp_val;
            if (s_val != 0.0f) {
                float scaled = x_val / s_val + zp_float;
                res = output_is_float_dtype ? (double)scaled : (double)rintf(x_val / s_val) + zp_val;
            }
        }
        set_quantize_linear_value(Y, i, res, saturate);
    }
}


// 实现 `quantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void quantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y) {
    quantize_linear_forward_impl(X, Scale, ZeroPoint, Y, 0, 1);
}


// 实现 `quantize linear` 的显式 precision 属性入口，用于 ONNX opset 25 的除法精度覆盖。
void quantize_linear_forward_precision(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y, int precision) {
    quantize_linear_forward_impl(X, Scale, ZeroPoint, Y, precision, 1);
}


// 实现 `quantize linear` 的 opset 25 属性入口，同时覆盖 precision 和 float8 saturate。
void quantize_linear_forward_precision_saturate(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y, int precision, int saturate) {
    quantize_linear_forward_impl(X, Scale, ZeroPoint, Y, precision, saturate);
}


// 实现 `dequantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dequantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y) {
    if (!X || !Scale || !ZeroPoint || !Y) return;

    size_t loop_size = Y->size;

    #pragma omp parallel for
    for (size_t i = 0; i < loop_size; i++) {
        // 1. 读取数据
        double x_val = get_value_as_double(X, i);
        double s_val = get_value_as_double(Scale, i);
        double zp_val = get_value_as_double(ZeroPoint, i);
    
        double res = (x_val - zp_val) * s_val;
        
        set_tensor_value_from_float(Y, i, res);
    }
}


// 实现 `gemm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gemm_forward(const Tensor* A, const Tensor* B, const Tensor* C, Tensor* Y, 
                  float alpha, float beta, int transA, int transB) {
    // 假设 A, B 已经是 2D 矩阵 (前端已处理 reshape)
    int M = (transA == 0) ? A->shape[0] : A->shape[1];
    int K = (transA == 0) ? A->shape[1] : A->shape[0];
    int N = (transB == 0) ? B->shape[1] : B->shape[0];
    
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            
            // 计算矩阵乘积: A' * B'
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                // 计算 A 的索引
                size_t idx_a = (transA == 0) ? ((size_t)m * A->shape[1] + k) 
                                             : ((size_t)k * A->shape[1] + m);
                
                // 计算 B 的索引
                size_t idx_b = (transB == 0) ? ((size_t)k * B->shape[1] + n) 
                                             : ((size_t)n * B->shape[1] + k);
                
                sum += get_value_as_double(A, idx_a) * get_value_as_double(B, idx_b);
            }
            
            double res = (double)alpha * sum;
            
            // 处理 Bias C
            if (C != NULL && C->data != NULL) {
                double val_c = 0.0;
                // 标量广播
                if (C->size == 1) {
                    val_c = get_value_as_double(C, 0);
                } 
                // 1D 张量处理 (通常是 (N,) 加在列上，或 (M,) 加在行上)
                else if (C->ndim == 1) {
                    if (C->shape[0] == N) {
                        val_c = get_value_as_double(C, n);
                    } 
                    else if (C->shape[0] == M) {
                        val_c = get_value_as_double(C, m);
                    }
                } 
                // 2D 及以上张量
                else if (C->ndim >= 2) {
                    int H = C->shape[C->ndim - 2]; // 倒数第二维
                    int W = C->shape[C->ndim - 1]; // 最后一维
                    int idx_h = (H == 1) ? 0 : m; 
                    int idx_w = (W == 1) ? 0 : n;

                    if (idx_h < H && idx_w < W) {
                        val_c = get_value_as_double(C, idx_h * W + idx_w);
                    }
                }
                res += (double)beta * val_c;
            }
            // 写入结果
            size_t y_idx = (size_t)m * N + n;
            set_tensor_value_from_float(Y, y_idx, res);
        }
    }
}


// 实现 `matmul integer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void matmul_integer_forward(const Tensor* A, const Tensor* B,
                            const Tensor* AZeroPoint, const Tensor* BZeroPoint,
                            Tensor* Y) {
    if (!A || !B || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM || ndim < 2) return;

    int K = A->shape[A->ndim - 1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, Y->shape, ndim);

        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        int64_t sum = 0;

        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim;
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;
                else if (d == A->ndim - 2) val = m;
                else {
                    int y_dim_idx = d + offset_a;
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += (size_t)val * stride_a;
                stride_a *= A->shape[d];
            }

            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;
                else if (d == B->ndim - 2) val = k;
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += (size_t)val * stride_b;
                stride_b *= B->shape[d];
            }

            int64_t a_val = get_value_as_int64(A, idx_a);
            int64_t b_val = get_value_as_int64(B, idx_b);
            int64_t a_zp = (AZeroPoint && AZeroPoint->data) ? get_value_as_int64(AZeroPoint, idx_a) : 0;
            int64_t b_zp = (BZeroPoint && BZeroPoint->data) ? get_value_as_int64(BZeroPoint, idx_b) : 0;
            sum += (a_val - a_zp) * (b_val - b_zp);
        }

        if (Y->dtype == DTYPE_INT32) {
            ((int32_t*)Y->data)[i] = (int32_t)sum;
        } else {
            set_tensor_value_from_int(Y, i, sum);
        }
    }
}


// 实现 `qlinear matmul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void qlinear_matmul_forward(const Tensor* A, const Tensor* AScale, const Tensor* AZeroPoint,
                            const Tensor* B, const Tensor* BScale, const Tensor* BZeroPoint,
                            const Tensor* YScale, const Tensor* YZeroPoint, Tensor* Y) {
    if (!A || !AScale || !AZeroPoint || !B || !BScale || !BZeroPoint || !YScale || !YZeroPoint || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM || ndim < 2) return;

    int K = A->shape[A->ndim - 1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, Y->shape, ndim);

        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        double acc = 0.0;

        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim;
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;
                else if (d == A->ndim - 2) val = m;
                else {
                    int y_dim_idx = d + offset_a;
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += (size_t)val * stride_a;
                stride_a *= A->shape[d];
            }

            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;
                else if (d == B->ndim - 2) val = k;
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += (size_t)val * stride_b;
                stride_b *= B->shape[d];
            }

            double a_real = (get_value_as_double(A, idx_a) - get_value_as_double(AZeroPoint, idx_a)) * get_value_as_double(AScale, idx_a);
            double b_real = (get_value_as_double(B, idx_b) - get_value_as_double(BZeroPoint, idx_b)) * get_value_as_double(BScale, idx_b);
            acc += a_real * b_real;
        }

        double y_scale = get_value_as_double(YScale, i);
        double y_zp = get_value_as_double(YZeroPoint, i);
        double q = y_zp;
        if (y_scale != 0.0) {
            q = nearbyint(acc / y_scale + y_zp);
        }
        set_tensor_value_from_float(Y, i, q);
    }
}


// DynamicQuantizeLinear
// 仅支持映射到 uint8 ([0, 255])
// 实现 `dynamic quantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dynamic_quantize_linear_forward(const Tensor* x, Tensor* y, Tensor* y_scale, Tensor* y_zp) {
    if (!x || !y || !y_scale || !y_zp) return;
    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;
    
    for (size_t i = 0; i < x->size; i++) {
        double val = get_value_as_double(x, i);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    min_val = fmin(min_val, 0.0);
    max_val = fmax(max_val, 0.0);
    
    // 计算 Scale 和 ZeroPoint
    // Q_max = 255, Q_min = 0
    double scale = (max_val - min_val) / 255.0;
    if (scale == 0.0) scale = 1.0; // 避免除以 0
    
    double zp_double = 0.0 - min_val / scale;
    // Saturate ZP to [0, 255]
    zp_double = round(zp_double);
    if (zp_double < 0.0) zp_double = 0.0;
    if (zp_double > 255.0) zp_double = 255.0;
    uint8_t zp = (uint8_t)zp_double;
    
    // 写入参数输出
    set_tensor_value_from_float(y_scale, 0, scale);
    // 直接写入 uint8 原始数据到 scalar tensor
    // 假设 y_zp 是 uint8 类型
    if (y_zp->dtype == DTYPE_UINT8) {
        ((uint8_t*)y_zp->data)[0] = zp;
    } else {
        set_tensor_value_from_float(y_zp, 0, (double)zp);
    }
    
    // 执行量化
    // y = saturate(round(x / scale) + zp)
    _Pragma("omp parallel for")
    for (size_t i = 0; i < x->size; i++) {
        double val = get_value_as_double(x, i);
        double q_val = rint(val / scale) + (double)zp;
        
        // Saturate to uint8
        if (q_val < 0.0) q_val = 0.0;
        if (q_val > 255.0) q_val = 255.0;
        
        // 写入
        // set_tensor_value 会根据 y 的类型 (uint8) 自动转换
        set_tensor_value_from_float(y, i, q_val);
    }
}
