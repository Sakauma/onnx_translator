/**
  ******************************************************************************
  * @file        tensor_ops_quantize_linear.c
  * @author      Egor Izmaylov
  * @brief       实现 QuantizeLinear 和 DequantizeLinear 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从矩阵/量化 shard 拆分线性量化实现。
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
