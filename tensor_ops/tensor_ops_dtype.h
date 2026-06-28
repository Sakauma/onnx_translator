/**
  ******************************************************************************
  * @file        tensor_ops_dtype.h
  * @author      Egor Izmaylov
  * @brief       Defines tensor dtype sizing, integer helpers, and low-precision float codecs.
  * @details     2026.06.28  V1.0.0  Created
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_DTYPE_H
#define TENSOR_OPS_DTYPE_H

#include "tensor_ops.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// 实现 `float_to_bits` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint32_t float_to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// 实现 `bits_to_float` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline float bits_to_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// 获取数据类型的字节大小
// 封装 `get_dtype_size` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT8_E4M3:
        case DTYPE_FLOAT8_E5M2:
        case DTYPE_FLOAT8_E4M3FNUZ:
        case DTYPE_FLOAT8_E5M2FNUZ:
        case DTYPE_FLOAT4_E2M1:
        case DTYPE_FLOAT8_E8M0:
        case DTYPE_BOOL:
        case DTYPE_INT4:
        case DTYPE_UINT4:
        case DTYPE_INT2:
        case DTYPE_UINT2:
        case DTYPE_INT8:
        case DTYPE_UINT8:
            return 1;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
        case DTYPE_INT16:
        case DTYPE_UINT16:
            return 2;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
        case DTYPE_UINT32:
            return 4;
        case DTYPE_COMPLEX64:
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_UINT64:
            return 8;
        case DTYPE_COMPLEX128:
            return 16;
        default:
            return 4;
    }
}

// 判断 dtype 是否属于整数族，包含 ONNX 支持的有符号和无符号整数。
static inline int is_integer_dtype(DataType dtype) {
    return dtype == DTYPE_INT4 ||
           dtype == DTYPE_UINT4 ||
           dtype == DTYPE_INT2 ||
           dtype == DTYPE_UINT2 ||
           dtype == DTYPE_INT8 ||
           dtype == DTYPE_UINT8 ||
           dtype == DTYPE_INT16 ||
           dtype == DTYPE_UINT16 ||
           dtype == DTYPE_INT32 ||
           dtype == DTYPE_UINT32 ||
           dtype == DTYPE_INT64 ||
           dtype == DTYPE_UINT64;
}

// 判断 dtype 是否属于无符号整数族，供比较、排序和回绕写回路径选择。
static inline int is_unsigned_integer_dtype(DataType dtype) {
    return dtype == DTYPE_UINT4 ||
           dtype == DTYPE_UINT2 ||
           dtype == DTYPE_UINT8 ||
           dtype == DTYPE_UINT16 ||
           dtype == DTYPE_UINT32 ||
           dtype == DTYPE_UINT64;
}

// 返回整数 dtype 的有效位宽，int4 按 4 位二补码整数处理。
static inline int integer_dtype_bits(DataType dtype) {
    switch (dtype) {
        case DTYPE_INT4: return 4;
        case DTYPE_UINT4: return 4;
        case DTYPE_INT2: return 2;
        case DTYPE_UINT2: return 2;
        case DTYPE_INT8:
        case DTYPE_UINT8: return 8;
        case DTYPE_INT16:
        case DTYPE_UINT16: return 16;
        case DTYPE_INT32:
        case DTYPE_UINT32: return 32;
        case DTYPE_INT64:
        case DTYPE_UINT64: return 64;
        default: return 0;
    }
}

// 将无符号位模式按目标有符号位宽解释，匹配 NumPy/ONNX Cast 的二补码 wrap 行为。
static inline int64_t sign_extend_integer_bits(uint64_t value, int bits) {
    if (bits >= 64) {
        int64_t result;
        memcpy(&result, &value, sizeof(result));
        return result;
    }
    uint64_t mask = (1ULL << bits) - 1ULL;
    uint64_t sign_bit = 1ULL << (bits - 1);
    value &= mask;
    if (value & sign_bit) {
        value |= ~mask;
    }
    return (int64_t)value;
}

// 将浮点数按“向零截断后按位宽取模”的规则转成无符号位模式，供 Cast 专用。
static inline uint64_t wrap_float_to_unsigned_bits(double value, int bits) {
    if (!isfinite(value) || bits <= 0) {
        return 0;
    }

    long double truncated = value < 0.0 ? ceill((long double)value) : floorl((long double)value);
    long double modulus = ldexpl(1.0L, bits);
    long double wrapped = fmodl(truncated, modulus);
    if (wrapped < 0.0L) {
        wrapped += modulus;
    }

    if (bits < 64) {
        uint64_t mask = (1ULL << bits) - 1ULL;
        return ((uint64_t)wrapped) & mask;
    }
    if (wrapped >= (long double)UINT64_MAX) {
        return UINT64_MAX;
    }
    return (uint64_t)wrapped;
}

// 4-bit 饱和截断
// 实现 `saturate_cast_int4` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline int8_t saturate_cast_int4(int64_t val) {
    if (val > 7) return 7;
    if (val < -8) return -8;
    return (int8_t)val;
}

// 4-bit 无符号饱和截断 (0 ~ 15)
// 实现 `saturate_cast_uint4` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint8_t saturate_cast_uint4(int64_t val) {
    if (val > 15) return 15;
    if (val < 0) return 0;
    return (uint8_t)val;
}

// 2-bit 饱和截断 (-2 ~ 1)
// 实现 `saturate_cast_int2` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline int8_t saturate_cast_int2(int64_t val) {
    if (val > 1) return 1;
    if (val < -2) return -2;
    return (int8_t)val;
}

// 2-bit 无符号饱和截断 (0 ~ 3)
// 实现 `saturate_cast_uint2` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint8_t saturate_cast_uint2(int64_t val) {
    if (val > 3) return 3;
    if (val < 0) return 0;
    return (uint8_t)val;
}

// 8-bit 饱和截断
// 实现 `saturate_cast_int8` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline int8_t saturate_cast_int8(int64_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// 8-bit 无符号饱和截断 (0 ~ 255)
// 实现 `saturate_cast_uint8` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint8_t saturate_cast_uint8(int64_t val) {
    if (val > 255) return 255;
    if (val < 0) return 0;
    return (uint8_t)val;
}

// 16-bit 无符号饱和截断 (0 ~ 65535)
// 实现 `saturate_cast_uint16` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint16_t saturate_cast_uint16(int64_t val) {
    if (val > 65535) return 65535;
    if (val < 0) return 0;
    return (uint16_t)val;
}

// 32-bit 无符号饱和截断 (0 ~ 4294967295)
// 实现 `saturate_cast_uint32` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint32_t saturate_cast_uint32(int64_t val) {
    if (val < 0) return 0;
    if ((uint64_t)val > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)val;
}

// 64-bit 无符号饱和截断，主要服务非 Cast 数值写回路径。
// 实现 `saturate_cast_uint64` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint64_t saturate_cast_uint64(int64_t val) {
    if (val < 0) return 0;
    return (uint64_t)val;
}

// 16-bit 饱和截断
// 实现 `saturate_cast_int16` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline int16_t saturate_cast_int16(int64_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// 32-bit 饱和截断
// 实现 `saturate_cast_int32` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline int32_t saturate_cast_int32(int64_t val) {
    if (val > 2147483647) return 2147483647;
    if (val < -2147483648) return -2147483648;
    return (int32_t)val;
}

/**
 * 将32位浮点数转换为16位浮点数
 *
 * @param value 32位浮点数
 * @return 16位浮点数
 */
// 实现 `float_to_float16` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint16_t float_to_float16(float value) {
    uint32_t bits = float_to_bits(value);
    uint16_t sign = (bits >> 16) & 0x8000;

    int32_t exp = ((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    if (exp == 255) {
        if (mant != 0) return sign | 0x7FFF; // NaN
        return sign | 0x7C00; // Inf
    }
    if (exp == 0 && mant == 0) return sign;
    int32_t new_exp = exp - 127 + 15;
    uint32_t full_mant = mant;
    if (exp != 0) {
        full_mant |= 0x800000; // 加上隐含的 1
    } else {
        new_exp++;
    }

    int shift = 13;
    if (new_exp <= 0) {
        shift += (1 - new_exp);
        new_exp = 0; // 编码指数设为 0
    }
    if (shift >= 24) return sign;

    uint32_t mant_10 = full_mant >> shift;
    uint32_t guard = (full_mant >> (shift - 1)) & 1;

    uint32_t mask = (1 << (shift - 1)) - 1;
    uint32_t sticky = (full_mant & mask) != 0;
    uint32_t lsb = mant_10 & 1;

    if (guard && (sticky || lsb)) {
        mant_10++;
        if (new_exp == 0) {
            if (mant_10 & 0x400) {
                new_exp = 1;
            }
        } else {
            if (mant_10 & 0x800) {
                mant_10 = 0;
                new_exp++;
            }
        }
    }
    if (new_exp >= 31) return sign | 0x7C00;
    return sign | (new_exp << 10) | (mant_10 & 0x3FF);
}

/**
 * 将16位浮点数转换为32位浮点数
 *
 * @param value 16位浮点数
 * @return 32位浮点数
 */
// 实现 `float16_to_float` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline float float16_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000) << 16;
    uint32_t exp  = (value >> 10) & 0x1F;
    uint32_t frac = value & 0x3FF;

    if (exp == 0 && frac == 0) {
        return bits_to_float(sign);
    }
    if (exp == 31) {
        return bits_to_float(sign | 0x7F800000 | (frac << 13));
    }
    if (exp == 0) {
        int32_t new_exp = -14 + 127;
        while ((frac & 0x400) == 0) {
            frac <<= 1;
            new_exp--;
        }
        frac &= 0x3FF;
        uint32_t bits = sign | (new_exp << 23) | (frac << 13);
        return bits_to_float(bits);
    }
    uint32_t new_exp = exp - 15 + 127;
    uint32_t bits = sign | (new_exp << 23) | (frac << 13);
    return bits_to_float(bits);
}

/**
 * 将32位浮点数转换为16位bfloat16格式
 *
 * @param value 32位浮点数
 * @return 16位bfloat16格式数据
 */
// 实现 `float_to_bfloat16` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint16_t float_to_bfloat16(float value) {
    uint32_t bits = float_to_bits(value);

    if ((bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) != 0) {
        return (uint16_t)(bits >> 16) | 0x0040; // 强制设为 Quiet NaN
    }

    uint32_t lsb    = (bits >> 16) & 1;
    uint32_t guard  = (bits >> 15) & 1;
    uint32_t sticky = (bits & 0x7FFF) != 0;
    uint32_t rnd = guard && (sticky || lsb);
    uint32_t rounded = bits + (rnd << 16);

    if ((bits & 0x7F800000) != 0x7F800000 && (rounded & 0x7F800000) == 0x7F800000) {
         // 保持符号，设为 Inf
        return (uint16_t)((bits & 0x80000000) >> 16) | 0x7F80;
    }
    return (uint16_t)(rounded >> 16);
}

/**
 * 将16位bfloat16格式数据转换为32位浮点数
 *
 * @param value 16位bfloat16格式数据
 * @return 32位浮点数
 */
// 实现 `bfloat16_to_float` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline float bfloat16_to_float(uint16_t value) {
    // 提取符号位
    uint32_t sign = ((uint32_t)(value & 0x8000)) << 16;
    // 提取指数位
    uint32_t exp = ((uint32_t)(value & 0x7F80)) << 16;
    // 提取尾数位
    uint32_t frac = ((uint32_t)(value & 0x007F)) << 16;
    // 组合符号位、指数位和尾数位
    uint32_t bits = sign | exp | frac;
    return bits_to_float(bits);
}

/**
 * 将8位float8_e4m3格式数据转换为32位浮点数
 *
 * @param value 8位float8_e4m3格式数据
 * @return 32位浮点数
 */
// 实现 `fp8_e4m3_to_float` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp  = (val & 0x78) >> 3;
    uint32_t mant = (val & 0x07);
    if (exp == 0 && mant == 0) return bits_to_float(sign);
    if (exp == 15 && mant == 7) {
        return bits_to_float(sign | 0x7F800000 | 0x400000);
    }

    if (exp == 0) {
        int32_t new_exp = -6 + 127;
        while ((mant & 0x08) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x07;
        return bits_to_float(sign | (new_exp << 23) | (mant << 20));
    }
    uint32_t new_exp = exp + 120;
    return bits_to_float(sign | (new_exp << 23) | (mant << 20));
}

// 实现 `float_to_fp8_e4m3` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint8_t float_to_fp8_e4m3(float f) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255 && mant != 0) return 0x7F | sign;
    if (exp == 0) return (uint8_t)sign;
    exp = exp - 127 + 7;
    if (exp < 1) return (uint8_t)sign;
    if (exp > 15) return 0x7E | sign;
    uint32_t mant_3 = (mant >> 20) & 0x7; // 截断后的尾数
    uint32_t guard  = (mant >> 19) & 1;
    uint32_t sticky = (mant & 0x7FFFF) != 0;
    uint32_t lsb    = mant_3 & 1;

    if (guard && (sticky || lsb)) {
        mant_3++;
        // 进位处理
        if (mant_3 > 7) {
            mant_3 = 0;
            exp++;
        }
    }
    if (exp > 15 || (exp == 15 && mant_3 == 7)) {
        return 0x7E | sign; // 饱和到最大值
    }

    return (uint8_t)(sign | (exp << 3) | mant_3);
}

/**
 * 将8位float8_e5m2格式数据转换为32位浮点数
 *
 * @param value 8位float8_e5m2格式数据
 * @return 32位浮点数
 */
// 实现 `fp8_e5m2_to_float` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp  = (val & 0x7C) >> 2;
    uint32_t mant = (val & 0x03);
    if (exp == 0 && mant == 0) return bits_to_float(sign);
    if (exp == 31) {
        uint32_t f32_mant = mant << 21;
        if (mant != 0) f32_mant |= 0x400000;
        return bits_to_float(sign | 0x7F800000 | f32_mant);
    }
    if (exp == 0) {
        int32_t new_exp = -14 + 127;
        while ((mant & 0x04) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x03;
        return bits_to_float(sign | (new_exp << 23) | (mant << 21));
    }
    uint32_t new_exp = exp + 112;
    return bits_to_float(sign | (new_exp << 23) | (mant << 21));
}

// 实现 `float_to_fp8_e5m2` 的数值格式转换或饱和裁剪，保证低精度存储符合 ONNX dtype 语义。
static inline uint8_t float_to_fp8_e5m2(float f) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255) {
        return (uint8_t)(sign | 0x7C | (mant ? 1 : 0));
    }
    if (exp == 0) return (uint8_t)sign;
    exp = exp - 127 + 15;
    if (exp < 1) return (uint8_t)sign;
    if (exp >= 31) return (uint8_t)(sign | 0x7C);

    uint32_t mant_2 = (mant >> 21) & 0x3;
    uint32_t guard  = (mant >> 20) & 1;
    uint32_t sticky = (mant & 0xFFFFF) != 0;
    uint32_t lsb    = mant_2 & 1;

    if (guard && (sticky || lsb)) {
        mant_2++;
        if (mant_2 > 3) {
            mant_2 = 0;
            exp++;
        }
    }
    if (exp >= 31) return (uint8_t)(sign | 0x7C);

    return (uint8_t)(sign | (exp << 2) | mant_2);
}

// 实现 `fp8_e4m3fnuz_to_float` 的 FNUZ 解码；0x80 是 NaN，零值不保留负号。
static inline float fp8_e4m3fnuz_to_float(uint8_t val) {
    if (val == 0x80) return bits_to_float(0x7FC00000);
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp = (val & 0x78) >> 3;
    uint32_t mant = val & 0x07;
    if (exp == 0 && mant == 0) return 0.0f;
    if (exp == 0) {
        int32_t new_exp = -7 + 127;
        while ((mant & 0x08) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x07;
        return bits_to_float(sign | ((uint32_t)new_exp << 23) | (mant << 20));
    }
    uint32_t new_exp = exp + 119;
    return bits_to_float(sign | (new_exp << 23) | (mant << 20));
}

// 实现 `fp8_e5m2fnuz_to_float` 的 FNUZ 解码；0x80 是 NaN，零值不保留负号。
static inline float fp8_e5m2fnuz_to_float(uint8_t val) {
    if (val == 0x80) return bits_to_float(0x7FC00000);
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp = (val & 0x7C) >> 2;
    uint32_t mant = val & 0x03;
    if (exp == 0 && mant == 0) return 0.0f;
    if (exp == 0) {
        int32_t new_exp = -15 + 127;
        while ((mant & 0x04) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x03;
        return bits_to_float(sign | ((uint32_t)new_exp << 23) | (mant << 21));
    }
    uint32_t new_exp = exp + 111;
    return bits_to_float(sign | (new_exp << 23) | (mant << 21));
}

// 将 float32 编码为 E4M3FNUZ；saturate=0 时溢出写入 FNUZ NaN。
static inline uint8_t float_to_fp8_e4m3fnuz_saturate(float f, int saturate) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255 && mant != 0) return 0x80;
    if (exp == 255) return saturate ? (uint8_t)(sign | 0x7F) : 0x80;
    if ((bits & 0x7FFFFFFF) == 0) return 0;

    double abs_value = fabs((double)f);
    int32_t target_exp = exp - 127 + 8;
    if (target_exp < 1) {
        uint32_t q = (uint32_t)rint(abs_value * 1024.0);
        if (q == 0) return 0;
        if (q >= 8) return (uint8_t)(sign | 0x08);
        return (uint8_t)(sign | q);
    }

    uint32_t mant_3 = (mant >> 20) & 0x7;
    uint32_t guard = (mant >> 19) & 1;
    uint32_t sticky = (mant & 0x7FFFF) != 0;
    uint32_t lsb = mant_3 & 1;
    if (guard && (sticky || lsb)) {
        mant_3++;
        if (mant_3 > 7) {
            mant_3 = 0;
            target_exp++;
        }
    }
    if (target_exp > 15) return saturate ? (uint8_t)(sign | 0x7F) : 0x80;
    return (uint8_t)(sign | ((uint32_t)target_exp << 3) | mant_3);
}

// 默认低精度写回采用饱和语义，与 ONNX Cast/QuantizeLinear 默认 saturate=1 对齐。
static inline uint8_t float_to_fp8_e4m3fnuz(float f) {
    return float_to_fp8_e4m3fnuz_saturate(f, 1);
}

// 将 float32 编码为 E5M2FNUZ；saturate=0 时溢出写入 FNUZ NaN。
static inline uint8_t float_to_fp8_e5m2fnuz_saturate(float f, int saturate) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits & 0x80000000) >> 24;
    int32_t exp = (int32_t)((bits & 0x7F800000) >> 23);
    uint32_t mant = bits & 0x007FFFFF;

    if (exp == 255 && mant != 0) return 0x80;
    if (exp == 255) return saturate ? (uint8_t)(sign | 0x7F) : 0x80;
    if ((bits & 0x7FFFFFFF) == 0) return 0;

    double abs_value = fabs((double)f);
    int32_t target_exp = exp - 127 + 16;
    if (target_exp < 1) {
        uint32_t q = (uint32_t)rint(abs_value * 131072.0);
        if (q == 0) return 0;
        if (q >= 4) return (uint8_t)(sign | 0x04);
        return (uint8_t)(sign | q);
    }

    uint32_t mant_2 = (mant >> 21) & 0x3;
    uint32_t guard = (mant >> 20) & 1;
    uint32_t sticky = (mant & 0xFFFFF) != 0;
    uint32_t lsb = mant_2 & 1;
    if (guard && (sticky || lsb)) {
        mant_2++;
        if (mant_2 > 3) {
            mant_2 = 0;
            target_exp++;
        }
    }
    if (target_exp > 31) return saturate ? (uint8_t)(sign | 0x7F) : 0x80;
    return (uint8_t)(sign | ((uint32_t)target_exp << 2) | mant_2);
}

// 默认低精度写回采用饱和语义，与 ONNX Cast/QuantizeLinear 默认 saturate=1 对齐。
static inline uint8_t float_to_fp8_e5m2fnuz(float f) {
    return float_to_fp8_e5m2fnuz_saturate(f, 1);
}

// 解码 ONNX FLOAT4E2M1 位模式；当前运行时使用 1 字节容器保存低 4 位。
static inline float fp4_e2m1_to_float(uint8_t val) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[val & 0x0F];
}

// 编码 ONNX FLOAT4E2M1，按最近偶数码值选择 16 个可表示值之一。
static inline uint8_t float_to_fp4_e2m1(float f) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    uint32_t bits = float_to_bits(f);
    if ((bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) != 0) return 0x08;
    if ((bits & 0x7FFFFFFF) == 0) return (uint8_t)((bits & 0x80000000) ? 0x08 : 0x00);
    int sign_group = (bits & 0x80000000) ? 8 : 0;
    uint8_t best_code = (uint8_t)sign_group;
    float best_diff = INFINITY;
    for (uint8_t offset = 0; offset < 8; ++offset) {
        uint8_t code = (uint8_t)(sign_group + offset);
        float diff = fabsf(f - table[code]);
        if (diff < best_diff || (diff == best_diff && ((code & 1) == 0))) {
            best_diff = diff;
            best_code = code;
        }
    }
    return best_code;
}

// 解码 ONNX FLOAT8E8M0FNU；0xFF 表示 NaN，其余编码表示 2^(code - 127)。
static inline float fp8_e8m0_to_float(uint8_t val) {
    if (val == 0xFF) return bits_to_float(0x7FC00000);
    return ldexpf(1.0f, (int)val - 127);
}

// 编码 ONNX FLOAT8E8M0FNU；非正数和 NaN/Inf 写入 NaN 编码，正有限数按 log2 最近整数编码。
static inline uint8_t float_to_fp8_e8m0(float f) {
    if (!isfinite(f) || f <= 0.0f) return 0xFF;
    int exp_code = (int)nearbyintf(log2f(f)) + 127;
    if (exp_code < 0) return 0x00;
    if (exp_code > 254) return 0xFE;
    return (uint8_t)exp_code;
}

#endif
