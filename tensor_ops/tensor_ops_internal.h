/**
  ******************************************************************************
  * @file        tensor_ops_internal.h
  * @author      Egor Izmaylov
  * @brief       声明 tensor_ops C 后端内部共享工具，包括 dtype 转换、Tensor 读写、坐标索引和宏模板。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_H
#define TENSOR_OPS_INTERNAL_H

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// tensor_ops/tensor_ops.c
#include "tensor_ops.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <float.h>

#define MAX_NDIM 16  

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// 余弦查找表大小
#define COS_LUT_SIZE 4096
// 余弦查找表位数
#define COS_LUT_BITS 12
// 余弦查找表
static double cos_lut[COS_LUT_SIZE + 1];
// 余弦查找表初始化标志
static int cos_lut_initialized = 0;
// 余弦查找表初始化互斥锁
static pthread_mutex_t cos_lut_mutex = PTHREAD_MUTEX_INITIALIZER;
// 圆周率常量
#define PI 3.141592653589793238462643383279502884197
// 两倍圆周率
#define TWO_PI (2.0 * PI)
// 半圆周率
#define HALF_PI (PI / 2.0)

static uint32_t simple_lcg(uint32_t* state);

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

// 用于排序
typedef struct {
    double value;
    uint64_t raw_value;
    int64_t signed_value;
    int64_t index;
} TopKElement;

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

/**
 * 创建张量
 * 
 * @param shape 张量形状数组
 * @param ndim 张量维度数
 * @param dtype 数据类型
 * @return 创建的张量指针
 */
// 实现 `create_tensor` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。

/**
 * 释放张量内存
 * 
 * @param tensor 要释放的张量指针
 */
// 实现 `free_tensor` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。

/*
 *
 * 从张量获取值，并作为 float 返回
 */
// 封装 `get_value_as_float` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline float get_value_as_float(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: return fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2: return fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E4M3FNUZ: return fp8_e4m3fnuz_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2FNUZ: return fp8_e5m2fnuz_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT4_E2M1: return fp4_e2m1_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E8M0: return fp8_e8m0_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT16: return float16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_BFLOAT16: return bfloat16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_FLOAT32: return ((float*)tensor->data)[index];
        case DTYPE_FLOAT64: return (float)((double*)tensor->data)[index];
        case DTYPE_INT4: {
            // INT4: 符号扩展到int8_t
            int8_t val = ((int8_t*)tensor->data)[index];
            // 确保符号位正确扩展
            if (val & 0x08) { // 检查第4位（符号位）
                val |= 0xF0;  // 符号扩展到8位
            } else {
                val &= 0x0F;  // 清除高位
            }
            return (float)val;
        }
        case DTYPE_UINT4: return (float)(((uint8_t*)tensor->data)[index] & 0x0F);
        case DTYPE_INT2: return (float)sign_extend_integer_bits(((uint8_t*)tensor->data)[index], 2);
        case DTYPE_UINT2: return (float)(((uint8_t*)tensor->data)[index] & 0x03);
        case DTYPE_INT8: return (float)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (float)((uint8_t*)tensor->data)[index];
        case DTYPE_BOOL: return ((uint8_t*)tensor->data)[index] ? 1.0f : 0.0f;
        case DTYPE_INT16: return (float)((int16_t*)tensor->data)[index];
        case DTYPE_UINT16: return (float)((uint16_t*)tensor->data)[index];
        case DTYPE_INT32: return (float)((int32_t*)tensor->data)[index];
        case DTYPE_UINT32: return (float)((uint32_t*)tensor->data)[index];
        case DTYPE_INT64: return (float)((int64_t*)tensor->data)[index];
        case DTYPE_UINT64: return (float)((uint64_t*)tensor->data)[index];
        default: return 0.0f;
    }
}

/*
 *
 * 从张量获取值，并作为 double 返回
 */
// 封装 `get_value_as_double` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline double get_value_as_double(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: return (double)fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2: return (double)fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E4M3FNUZ: return (double)fp8_e4m3fnuz_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2FNUZ: return (double)fp8_e5m2fnuz_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT4_E2M1: return (double)fp4_e2m1_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E8M0: return (double)fp8_e8m0_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT32: return (double)((float*)tensor->data)[index];
        case DTYPE_FLOAT16: return (double)float16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_BFLOAT16: return (double)bfloat16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_INT4: {
            // INT4: 符号扩展到int8_t
            int8_t val = ((int8_t*)tensor->data)[index];
            // 确保符号位正确扩展
            if (val & 0x08) { // 检查第4位（符号位）
                val |= 0xF0;  // 符号扩展到8位
            } else {
                val &= 0x0F;  // 清除高位
            }
            return (double)val;
        }
        case DTYPE_UINT4: return (double)(((uint8_t*)tensor->data)[index] & 0x0F);
        case DTYPE_INT2: return (double)sign_extend_integer_bits(((uint8_t*)tensor->data)[index], 2);
        case DTYPE_UINT2: return (double)(((uint8_t*)tensor->data)[index] & 0x03);
        case DTYPE_INT8: return (double)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (double)((uint8_t*)tensor->data)[index];
        case DTYPE_BOOL: return ((uint8_t*)tensor->data)[index] ? 1.0 : 0.0;
        case DTYPE_INT16: return (double)((int16_t*)tensor->data)[index];
        case DTYPE_UINT16: return (double)((uint16_t*)tensor->data)[index];
        case DTYPE_INT32: return (double)((int32_t*)tensor->data)[index];
        case DTYPE_UINT32: return (double)((uint32_t*)tensor->data)[index];
        case DTYPE_INT64: return (double)((int64_t*)tensor->data)[index];
        case DTYPE_UINT64: return (double)((uint64_t*)tensor->data)[index];
        case DTYPE_FLOAT64: return ((double*)tensor->data)[index];
        default: return 0.0;
    }
}

/*
 *
 * 从张量获取值，并作为 int64_t 返回
 */
// 封装 `get_value_as_int64` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline int64_t get_value_as_int64(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT32: return (int64_t)rintf(((float*)tensor->data)[index]);
        case DTYPE_FLOAT16: return (int64_t)rintf(float16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_BFLOAT16: return (int64_t)rintf(bfloat16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E4M3: return (int64_t)rintf(fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E5M2: return (int64_t)rintf(fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E4M3FNUZ: return (int64_t)rintf(fp8_e4m3fnuz_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E5M2FNUZ: return (int64_t)rintf(fp8_e5m2fnuz_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT4_E2M1: return (int64_t)rintf(fp4_e2m1_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E8M0: return (int64_t)rintf(fp8_e8m0_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_INT4: {
            // INT4: 符号扩展到int8_t
            int8_t val = ((int8_t*)tensor->data)[index];
            // 确保符号位正确扩展
            if (val & 0x08) { // 检查第4位（符号位）
                val |= 0xF0;  // 符号扩展到8位
            } else {
                val &= 0x0F;  // 清除高位
            }
            return (int64_t)val;
        }
        case DTYPE_UINT4: return (int64_t)(((uint8_t*)tensor->data)[index] & 0x0F);
        case DTYPE_INT2: return sign_extend_integer_bits(((uint8_t*)tensor->data)[index], 2);
        case DTYPE_UINT2: return (int64_t)(((uint8_t*)tensor->data)[index] & 0x03);
        case DTYPE_INT8: return (int64_t)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (int64_t)((uint8_t*)tensor->data)[index];
        case DTYPE_BOOL: return ((uint8_t*)tensor->data)[index] ? 1 : 0;
        case DTYPE_INT16: return (int64_t)((int16_t*)tensor->data)[index];
        case DTYPE_UINT16: return (int64_t)((uint16_t*)tensor->data)[index];
        case DTYPE_INT32: return (int64_t)((int32_t*)tensor->data)[index];
        case DTYPE_UINT32: return (int64_t)((uint32_t*)tensor->data)[index];
        case DTYPE_INT64: return ((int64_t*)tensor->data)[index];
        case DTYPE_UINT64: {
            uint64_t value = ((uint64_t*)tensor->data)[index];
            return value > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)value;
        }
        case DTYPE_FLOAT64: return (int64_t)rint(((double*)tensor->data)[index]);
        default: return 0;
    }
}

// 按整数 Cast 语义读取源元素的底层无符号位模式；有符号负数会自然映射到二补码表示。
static inline uint64_t get_integer_value_as_uint64(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_INT4: {
            int8_t val = ((int8_t*)tensor->data)[index];
            if (val & 0x08) {
                val |= 0xF0;
            } else {
                val &= 0x0F;
            }
            return (uint64_t)(int64_t)val;
        }
        case DTYPE_UINT4: return (uint64_t)(((uint8_t*)tensor->data)[index] & 0x0F);
        case DTYPE_INT2: return (uint64_t)sign_extend_integer_bits(((uint8_t*)tensor->data)[index], 2);
        case DTYPE_UINT2: return (uint64_t)(((uint8_t*)tensor->data)[index] & 0x03);
        case DTYPE_INT8: return (uint64_t)(int64_t)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (uint64_t)((uint8_t*)tensor->data)[index];
        case DTYPE_BOOL: return ((uint8_t*)tensor->data)[index] ? 1ULL : 0ULL;
        case DTYPE_INT16: return (uint64_t)(int64_t)((int16_t*)tensor->data)[index];
        case DTYPE_UINT16: return (uint64_t)((uint16_t*)tensor->data)[index];
        case DTYPE_INT32: return (uint64_t)(int64_t)((int32_t*)tensor->data)[index];
        case DTYPE_UINT32: return (uint64_t)((uint32_t*)tensor->data)[index];
        case DTYPE_INT64: return (uint64_t)((int64_t*)tensor->data)[index];
        case DTYPE_UINT64: return ((uint64_t*)tensor->data)[index];
        default: return wrap_float_to_unsigned_bits(get_value_as_double(tensor, index), 64);
    }
}

typedef enum {
    TENSOR_COMPARE_EQ = 0,
    TENSOR_COMPARE_GT = 1,
    TENSOR_COMPARE_LT = 2,
    TENSOR_COMPARE_GE = 3,
    TENSOR_COMPARE_LE = 4,
} TensorCompareOp;

// 精确比较两个整数元素，避免 int64/uint64 在 double 路径中丢失相邻大整数的低位。
static inline int compare_integer_values_exact(const Tensor* A, size_t a_index, const Tensor* B, size_t b_index) {
    int a_unsigned = is_unsigned_integer_dtype(A->dtype);
    int b_unsigned = is_unsigned_integer_dtype(B->dtype);

    if (a_unsigned && b_unsigned) {
        uint64_t a = get_integer_value_as_uint64(A, a_index);
        uint64_t b = get_integer_value_as_uint64(B, b_index);
        return (a > b) - (a < b);
    }

    if (!a_unsigned && !b_unsigned) {
        int64_t a = get_value_as_int64(A, a_index);
        int64_t b = get_value_as_int64(B, b_index);
        return (a > b) - (a < b);
    }

    if (a_unsigned) {
        uint64_t a = get_integer_value_as_uint64(A, a_index);
        int64_t b = get_value_as_int64(B, b_index);
        if (b < 0) return 1;
        uint64_t b_u = (uint64_t)b;
        return (a > b_u) - (a < b_u);
    }

    int64_t a = get_value_as_int64(A, a_index);
    uint64_t b = get_integer_value_as_uint64(B, b_index);
    if (a < 0) return -1;
    uint64_t a_u = (uint64_t)a;
    return (a_u > b) - (a_u < b);
}

// 根据比较关系返回布尔结果，整数路径使用精确比较，其他 dtype 保持原有 double 语义。
static inline int compare_tensor_values(const Tensor* A, size_t a_index, const Tensor* B, size_t b_index, TensorCompareOp op) {
    if (is_integer_dtype(A->dtype) && is_integer_dtype(B->dtype)) {
        int cmp = compare_integer_values_exact(A, a_index, B, b_index);
        switch (op) {
            case TENSOR_COMPARE_EQ: return cmp == 0;
            case TENSOR_COMPARE_GT: return cmp > 0;
            case TENSOR_COMPARE_LT: return cmp < 0;
            case TENSOR_COMPARE_GE: return cmp >= 0;
            case TENSOR_COMPARE_LE: return cmp <= 0;
            default: return 0;
        }
    }

    double a = get_value_as_double(A, a_index);
    double b = get_value_as_double(B, b_index);
    switch (op) {
        case TENSOR_COMPARE_EQ: return a == b;
        case TENSOR_COMPARE_GT: return a > b;
        case TENSOR_COMPARE_LT: return a < b;
        case TENSOR_COMPARE_GE: return a >= b;
        case TENSOR_COMPARE_LE: return a <= b;
        default: return 0;
    }
}

// 按目标整数 dtype 的位宽写入底层位模式；signed 目标按二补码解释，unsigned 目标自然截断。
static inline void set_integer_value_wrapped(Tensor* tensor, size_t index, uint64_t raw_value) {
    switch (tensor->dtype) {
        case DTYPE_INT4:   ((int8_t*)tensor->data)[index] = (int8_t)sign_extend_integer_bits(raw_value, 4); break;
        case DTYPE_UINT4:  ((uint8_t*)tensor->data)[index] = (uint8_t)(raw_value & 0x0F); break;
        case DTYPE_INT2:   ((int8_t*)tensor->data)[index] = (int8_t)sign_extend_integer_bits(raw_value, 2); break;
        case DTYPE_UINT2:  ((uint8_t*)tensor->data)[index] = (uint8_t)(raw_value & 0x03); break;
        case DTYPE_INT8:   ((int8_t*)tensor->data)[index] = (int8_t)sign_extend_integer_bits(raw_value, 8); break;
        case DTYPE_UINT8:  ((uint8_t*)tensor->data)[index] = (uint8_t)raw_value; break;
        case DTYPE_BOOL:   ((uint8_t*)tensor->data)[index] = raw_value != 0; break;
        case DTYPE_INT16:  ((int16_t*)tensor->data)[index] = (int16_t)sign_extend_integer_bits(raw_value, 16); break;
        case DTYPE_UINT16: ((uint16_t*)tensor->data)[index] = (uint16_t)raw_value; break;
        case DTYPE_INT32:  ((int32_t*)tensor->data)[index] = (int32_t)sign_extend_integer_bits(raw_value, 32); break;
        case DTYPE_UINT32: ((uint32_t*)tensor->data)[index] = (uint32_t)raw_value; break;
        case DTYPE_INT64:  ((int64_t*)tensor->data)[index] = sign_extend_integer_bits(raw_value, 64); break;
        case DTYPE_UINT64: ((uint64_t*)tensor->data)[index] = raw_value; break;
        default: break;
    }
}

/* 
 * 通用写入函数
 * 负责将计算结果安全地写入输出张量
 */
// 封装 `set_tensor_value_from_int` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline void set_tensor_value_from_int(Tensor* tensor, size_t index, int64_t value) {
    switch (tensor->dtype) {
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = saturate_cast_int4(value); break;
        case DTYPE_UINT4:   ((uint8_t*)tensor->data)[index] = saturate_cast_uint4(value); break;
        case DTYPE_INT2:    ((int8_t*)tensor->data)[index] = saturate_cast_int2(value); break;
        case DTYPE_UINT2:   ((uint8_t*)tensor->data)[index] = saturate_cast_uint2(value); break;
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = saturate_cast_int8(value); break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = saturate_cast_uint8(value); break;
        case DTYPE_BOOL:    ((uint8_t*)tensor->data)[index] = value != 0; break;
        case DTYPE_INT16:   ((int16_t*)tensor->data)[index] = saturate_cast_int16(value); break;
        case DTYPE_UINT16:  ((uint16_t*)tensor->data)[index] = saturate_cast_uint16(value); break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = saturate_cast_int32(value); break;
        case DTYPE_UINT32:  ((uint32_t*)tensor->data)[index] = saturate_cast_uint32(value); break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = value; break;
        case DTYPE_UINT64:  ((uint64_t*)tensor->data)[index] = saturate_cast_uint64(value); break;
        // 如果目标是浮点，进行转换
        case DTYPE_FLOAT8_E4M3: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3((float)value); break;
        case DTYPE_FLOAT8_E5M2: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2((float)value); break;
        case DTYPE_FLOAT8_E4M3FNUZ: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3fnuz((float)value); break;
        case DTYPE_FLOAT8_E5M2FNUZ: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2fnuz((float)value); break;
        case DTYPE_FLOAT4_E2M1: ((uint8_t*)tensor->data)[index] = float_to_fp4_e2m1((float)value); break;
        case DTYPE_FLOAT8_E8M0: ((uint8_t*)tensor->data)[index] = float_to_fp8_e8m0((float)value); break;
        case DTYPE_FLOAT16:     ((uint16_t*)tensor->data)[index] = float_to_float16((float)value); break;
        case DTYPE_BFLOAT16:    ((uint16_t*)tensor->data)[index] = float_to_bfloat16((float)value); break;
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = (double)value; break;
        default: break;
    }
}

// 封装 `set_tensor_value_from_float` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline void set_tensor_value_from_float(Tensor* tensor, size_t index, double value) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3((float)value); break;
        case DTYPE_FLOAT8_E5M2: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2((float)value); break;
        case DTYPE_FLOAT8_E4M3FNUZ: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3fnuz((float)value); break;
        case DTYPE_FLOAT8_E5M2FNUZ: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2fnuz((float)value); break;
        case DTYPE_FLOAT4_E2M1: ((uint8_t*)tensor->data)[index] = float_to_fp4_e2m1((float)value); break;
        case DTYPE_FLOAT8_E8M0: ((uint8_t*)tensor->data)[index] = float_to_fp8_e8m0((float)value); break;
        case DTYPE_FLOAT16:  ((uint16_t*)tensor->data)[index] = float_to_float16((float)value); break;
        case DTYPE_BFLOAT16: ((uint16_t*)tensor->data)[index] = float_to_bfloat16((float)value); break;
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = value; break;
        // 如果目标是整数，使用饱和截断转换
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = saturate_cast_int4((int64_t)rint(value)); break; 
        case DTYPE_UINT4:   ((uint8_t*)tensor->data)[index] = saturate_cast_uint4((int64_t)rint(value)); break;
        case DTYPE_INT2:    ((int8_t*)tensor->data)[index] = saturate_cast_int2((int64_t)rint(value)); break;
        case DTYPE_UINT2:   ((uint8_t*)tensor->data)[index] = saturate_cast_uint2((int64_t)rint(value)); break;
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = saturate_cast_int8((int64_t)rint(value)); break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = saturate_cast_uint8((int64_t)rint(value)); break;
        case DTYPE_BOOL:    ((uint8_t*)tensor->data)[index] = value != 0.0; break;
        case DTYPE_INT16:   ((int16_t*)tensor->data)[index] = saturate_cast_int16((int64_t)rint(value)); break;
        case DTYPE_UINT16:  ((uint16_t*)tensor->data)[index] = saturate_cast_uint16((int64_t)rint(value)); break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = saturate_cast_int32((int64_t)rint(value)); break;
        case DTYPE_UINT32:  ((uint32_t*)tensor->data)[index] = saturate_cast_uint32((int64_t)rint(value)); break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = (int64_t)rint(value); break;
        case DTYPE_UINT64:  ((uint64_t*)tensor->data)[index] = saturate_cast_uint64((int64_t)rint(value)); break;
        default: break;
    }
}

// 按 Cast 语义写入整数目标：整数间转换使用 modulo/wrap，区别于量化和普通算子的饱和写回。
static inline void set_tensor_value_for_cast(Tensor* tensor, size_t index, const Tensor* input, size_t input_index) {
    if (!tensor || !input || !tensor->data || !input->data) return;

    if (tensor->dtype == DTYPE_BOOL) {
        set_tensor_value_from_int(tensor, index, get_value_as_double(input, input_index) != 0.0);
        return;
    }

    if (!is_integer_dtype(tensor->dtype)) {
        set_tensor_value_from_float(tensor, index, get_value_as_double(input, input_index));
        return;
    }

    int bits = integer_dtype_bits(tensor->dtype);
    uint64_t raw_value = is_integer_dtype(input->dtype)
        ? get_integer_value_as_uint64(input, input_index)
        : wrap_float_to_unsigned_bits(get_value_as_double(input, input_index), bits);

    set_integer_value_wrapped(tensor, index, raw_value);
}

// 封装 `copy_tensor_element` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline void copy_tensor_element(Tensor* dst, size_t dst_index, const Tensor* src, size_t src_index) {
    if (!dst || !src || !dst->data || !src->data) return;

    size_t dst_elem_size = get_dtype_size(dst->dtype);
    size_t src_elem_size = get_dtype_size(src->dtype);
    if (dst->dtype == src->dtype && dst_elem_size == src_elem_size) {
        memcpy((uint8_t*)dst->data + dst_index * dst_elem_size,
               (const uint8_t*)src->data + src_index * src_elem_size,
               dst_elem_size);
        return;
    }

    set_tensor_value_from_float(dst, dst_index, get_value_as_double(src, src_index));
}

// Scatter 类算子共用的写入逻辑：none 直接复制元素，整数 reduction 保持 dtype 位宽回绕。
static inline void apply_scatter_update(Tensor* data, size_t data_index, const Tensor* updates, size_t update_index, int reduction) {
    if (!data || !updates || !data->data || !updates->data) return;

    if (reduction == 0) {
        copy_tensor_element(data, data_index, updates, update_index);
        return;
    }

    if (is_integer_dtype(data->dtype) && is_integer_dtype(updates->dtype)) {
        uint64_t old_value = get_integer_value_as_uint64(data, data_index);
        uint64_t update_value = get_integer_value_as_uint64(updates, update_index);
        uint64_t result = reduction == 1 ? old_value + update_value : old_value * update_value;
        set_integer_value_wrapped(data, data_index, result);
        return;
    }

    double old_value = get_value_as_double(data, data_index);
    double update_value = get_value_as_double(updates, update_index);
    double result = reduction == 1 ? old_value + update_value : old_value * update_value;
    set_tensor_value_from_float(data, data_index, result);
}

/* 判断是否为整数类型 */
#define IS_INT_TYPE(d) is_integer_dtype(d)

// --- 通用一元算子宏模板 ---
#ifndef UNARY_OP_IMPL
// 展开 `UNARY_OP_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define UNARY_OP_IMPL(FUNC_NAME, MATH_LOGIC) \
void FUNC_NAME(const Tensor* input, Tensor* output) { \
    if (!input || !output || !input->data || !output->data || input->size != output->size) return; \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < input->size; i++) { \
        double val = get_value_as_double(input, i); \
        double res = MATH_LOGIC; \
        set_tensor_value_from_float(output, i, res); \
    } \
}
#endif

/* 
   OP_FUNC: 执行计算的逻辑 (a + b, a - b 等)
*/
// 展开 `BINARY_OP_INT_LOGIC` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define BINARY_OP_INT_LOGIC(OP_FUNC) \
    switch (O->dtype) { \
        case DTYPE_INT32: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                set_integer_value_wrapped(O, i, (uint64_t)res); \
            } \
            break; \
        } \
        case DTYPE_INT16: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                set_integer_value_wrapped(O, i, (uint64_t)res); \
            } \
            break; \
        } \
        case DTYPE_INT8: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                set_integer_value_wrapped(O, i, (uint64_t)res); \
            } \
            break; \
        } \
        case DTYPE_UINT2: \
        case DTYPE_UINT4: \
        case DTYPE_UINT8: \
        case DTYPE_UINT16: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                uint64_t val_a = get_integer_value_as_uint64(A, i); \
                uint64_t val_b = get_integer_value_as_uint64(B, i); \
                uint64_t res = OP_FUNC##_u(val_a, val_b); \
                set_integer_value_wrapped(O, i, res); \
            } \
            break; \
        } \
        case DTYPE_UINT32: \
        case DTYPE_UINT64: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                uint64_t val_a = get_integer_value_as_uint64(A, i); \
                uint64_t val_b = get_integer_value_as_uint64(B, i); \
                uint64_t res = OP_FUNC##_u(val_a, val_b); \
                set_integer_value_wrapped(O, i, res); \
            } \
            break; \
        } \
        case DTYPE_INT2: \
        case DTYPE_INT4: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                set_integer_value_wrapped(O, i, (uint64_t)res); \
            } \
            break; \
        } \
        case DTYPE_INT64: { \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                set_integer_value_wrapped(O, i, (uint64_t)res); \
            } \
            break; \
        } \
        default: break; \
    }

// 简单的运算包装器，用于宏
// 实现 `op_add` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_add(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a + (uint64_t)b, 64); }
// 实现 `op_add_u` 的无符号整数版本，输出由目标 dtype 位宽自然回绕。
static inline uint64_t op_add_u(uint64_t a, uint64_t b) { return a + b; }
// 实现 `op_sub` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_sub(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a - (uint64_t)b, 64); }
// 实现 `op_sub_u` 的无符号整数版本，输出由目标 dtype 位宽自然回绕。
static inline uint64_t op_sub_u(uint64_t a, uint64_t b) { return a - b; }
// 实现 `op_mul` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_mul(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a * (uint64_t)b, 64); }
// 实现 `op_mul_u` 的无符号整数版本，输出由目标 dtype 位宽自然回绕。
static inline uint64_t op_mul_u(uint64_t a, uint64_t b) { return a * b; }
// 实现 `op_div` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_div(int64_t a, int64_t b) {
    if (b == 0) return a >= 0 ? INT64_MAX : INT64_MIN;
    if (a == INT64_MIN && b == -1) return INT64_MIN;
    return a / b;
}
// 实现 `op_div_u` 的无符号整数版本；除零保持既有保护策略，避免 C 未定义行为。
static inline uint64_t op_div_u(uint64_t a, uint64_t b) { return b == 0 ? UINT64_MAX : a / b; }

// 安全获取4D张量的值
// 封装 `get_val_4d_with_padding` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline double get_val_4d_with_padding(const Tensor* T, int n, int c, int h, int w, double pad_val) {
    int N = T->shape[0];
    int C = T->shape[1];
    int H = T->shape[2];
    int W = T->shape[3];

    // 越界检查：如果坐标在张量范围外，返回 padding 值
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) {
        return pad_val;
    }
    // 计算平坦索引
    size_t idx = ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)h * W) + w;
    return get_value_as_double(T, idx);
}

/**
 * ReLU激活函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `relu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * Abs函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `abs` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * 初始化余弦查找表
 * 使用泰勒级数展开计算余弦值并存储在查找表中
 */
// 实现 `init_cos_lut` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。

/**
 * 使用查找表计算余弦值
 * 
 * @param x 输入角度（弧度）
 * @return 余弦值
 */
// 实现 `cos_lut_lookup` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double cos_lut_lookup(double x) {
    // 如果查找表未初始化，则先初始化
    if (!cos_lut_initialized) {
        init_cos_lut();
    }
    // 处理负角度并归一化到[0, 2π]区间
    double reduced = fmod(fabs(x), TWO_PI);
    // 计算查找表索引和插值因子
    double idx_f = reduced * COS_LUT_SIZE / TWO_PI;
    int idx = (int)idx_f;
    double frac = idx_f - idx;
    // 边界处理
    if (idx >= COS_LUT_SIZE) {
        idx = COS_LUT_SIZE - 1;
        frac = 0.0;
    }
    // 线性插值计算余弦值
    return cos_lut[idx] * (1.0 - frac) + cos_lut[idx + 1] * frac;
}

/**
 * 余弦函数前向传播
 * * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `cos` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * Add函数前向传播实现
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `add` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * Sub函数前向传播实现 (A - B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `sub` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * Mul函数前向传播实现 (A * B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `mul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

/**
 * Div函数前向传播实现 (A / B)
 * 
 * 假设: A, B, 和 O 具有完全相同的形状 (广播已在Python层处理)
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量 (决定了计算精度)
 */
// 实现 `div` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `quantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `dequantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `conv2d` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `conv transpose2d` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `conv integer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `qlinear conv` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `max pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `max unpool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `max roi pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `roi_align_bilinear_sample` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double roi_align_bilinear_sample(const Tensor* X, int batch, int channel, double y, double x) {
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (y < -1.0 || y > (double)height || x < -1.0 || x > (double)width) {
        return 0.0;
    }
    if (y < 0.0) y = 0.0;
    if (x < 0.0) x = 0.0;
    int y0 = (int)y;
    int x0 = (int)x;
    int y1;
    int x1;
    if (y0 >= height - 1) {
        y1 = y0 = height - 1;
        y = (double)y0;
    } else {
        y1 = y0 + 1;
    }
    if (x0 >= width - 1) {
        x1 = x0 = width - 1;
        x = (double)x0;
    } else {
        x1 = x0 + 1;
    }
    double ly = y - (double)y0;
    double lx = x - (double)x0;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    double total = 0.0;
    int ys[2] = {y0, y1};
    int xs[2] = {x0, x1};
    double wy[2] = {hy, ly};
    double wx[2] = {hx, lx};
    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            size_t idx = ((size_t)batch * channels * height * width)
                       + ((size_t)channel * height * width)
                       + ((size_t)ys[iy] * width)
                       + (size_t)xs[ix];
            total += get_value_as_double(X, idx) * wy[iy] * wx[ix];
        }
    }
    return total;
}

// 实现 `roi_align_max_weighted_term` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double roi_align_max_weighted_term(const Tensor* X, int batch, int channel, double y, double x) {
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (y < -1.0 || y > (double)height || x < -1.0 || x > (double)width) {
        return 0.0;
    }
    if (y < 0.0) y = 0.0;
    if (x < 0.0) x = 0.0;
    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;
    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (double)y_low;
    } else {
        y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (double)x_low;
    } else {
        x_high = x_low + 1;
    }
    double ly = y - (double)y_low;
    double lx = x - (double)x_low;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    int ys[2] = {y_low, y_high};
    int xs[2] = {x_low, x_high};
    double wy[2] = {hy, ly};
    double wx[2] = {hx, lx};
    double max_term = -DBL_MAX;
    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            size_t idx = ((size_t)batch * channels * height * width)
                       + ((size_t)channel * height * width)
                       + ((size_t)ys[iy] * width)
                       + (size_t)xs[ix];
            double term = get_value_as_double(X, idx) * wy[iy] * wx[ix];
            if (term > max_term) max_term = term;
        }
    }
    return max_term;
}

// 实现 `roi align` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `gemm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ================== Softmax 实现 ==================
// 实现 `softmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Exp 实现

// Log 实现
// 未需要处理 log(0) 或负数的情况

// Sqrt 实现

// Sigmoid 实现

// Tanh 实现

// Flatten 实现
// 实现 `flatten` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Reshape 实现
// 实现 `reshape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 从平坦索引反解 N 维坐标
// 封装 `get_coords_from_index` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline void get_coords_from_index(size_t index, int* coords, int* shape, int ndim) {
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = index % shape[i];
        index /= shape[i];
    }
}

// 从 N 维坐标计算平坦索引
// 封装 `get_index_from_coords` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static inline size_t get_index_from_coords(int* coords, int* shape, int ndim) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        index += coords[i] * stride;
        stride *= shape[i];
    }
    return index;
}

// Transpose 实现
// 实现 `transpose` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 整数辅助函数
// 实现 `op_max` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_max(int64_t a, int64_t b) { return a > b ? a : b; }
// 实现 `op_max_u` 的无符号整数版本，按无符号大小关系比较。
static inline uint64_t op_max_u(uint64_t a, uint64_t b) { return a > b ? a : b; }
// 实现 `op_min` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_min(int64_t a, int64_t b) { return a < b ? a : b; }
// 实现 `op_min_u` 的无符号整数版本，按无符号大小关系比较。
static inline uint64_t op_min_u(uint64_t a, uint64_t b) { return a < b ? a : b; }

// Pow 实现
// 实现 `pow` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Max 实现
// 实现 `max` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Min 实现
// 实现 `min` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `concat` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `slice` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Neg

// Reciprocal

// Ceil

// Floor

// Cast
// 读取时自动转 double，写入 set_tensor_value 时会自动转为 output->dtype
// 实现 `cast` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `sum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `prelu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `det` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `tensor_scalar_equal` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int tensor_scalar_equal(const Tensor* tensor, size_t lhs, size_t rhs) {
    if (!tensor || !tensor->data) return 0;
    if (IS_INT_TYPE(tensor->dtype)) {
        return get_value_as_int64(tensor, lhs) == get_value_as_int64(tensor, rhs);
    }
    double a = get_value_as_double(tensor, lhs);
    double b = get_value_as_double(tensor, rhs);
    if (isnan(a) && isnan(b)) return 1;
    return a == b;
}

// 作为 `tensor_scalar_compare` 排序比较函数，保证排序类算子的值和索引顺序稳定。
static int tensor_scalar_compare(const Tensor* tensor, size_t lhs, size_t rhs) {
    if (IS_INT_TYPE(tensor->dtype)) {
        int64_t a = get_value_as_int64(tensor, lhs);
        int64_t b = get_value_as_int64(tensor, rhs);
        return (a > b) - (a < b);
    }
    double a = get_value_as_double(tensor, lhs);
    double b = get_value_as_double(tensor, rhs);
    int a_nan = isnan(a);
    int b_nan = isnan(b);
    if (a_nan && b_nan) return 0;
    if (a_nan) return 1;
    if (b_nan) return -1;
    return (a > b) - (a < b);
}

// 实现 `unique` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `hz_to_mel` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double hz_to_mel(double frequency) {
    return 2595.0 * log10(1.0 + frequency / 700.0);
}

// 实现 `mel_to_hz` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// 实现 `mel weight matrix` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `complex_tensor_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t complex_tensor_index(const Tensor* tensor, const int* coords, int component) {
    int complex_rank = tensor->ndim - 1;
    size_t idx = 0;
    for (int d = 0; d < complex_rank; d++) {
        idx = idx * (size_t)tensor->shape[d] + (size_t)coords[d];
    }
    return idx * (size_t)tensor->shape[complex_rank] + (size_t)component;
}

// 封装 `get_complex_value` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static void get_complex_value(const Tensor* tensor, const int* coords, double* real, double* imag) {
    *real = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 0));
    *imag = 0.0;
    if (tensor->shape[tensor->ndim - 1] == 2) {
        *imag = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 1));
    }
}

// 实现 `normalize_complex_axis` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int normalize_complex_axis(int axis, int complex_rank) {
    if (axis < 0) axis += complex_rank + 1;
    return axis;
}

// 实现 `dft` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `stft` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `recurrent_alpha` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_alpha(const float* values, int index, double default_value) {
    if (!values) return default_value;
    float value = values[index];
    return isnan(value) ? default_value : (double)value;
}

// 实现 `recurrent_clip` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_clip(double value, float clip, int has_clip) {
    if (!has_clip) return value;
    if (value > (double)clip) return (double)clip;
    if (value < -(double)clip) return -(double)clip;
    return value;
}

// 实现 `recurrent_activation` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_activation(double x, int code, const float* alphas, const float* betas, int index) {
    switch (code) {
        case 1:
            return 1.0 / (1.0 + exp(-x));
        case 2:
            return x > 0.0 ? x : 0.0;
        case 3: {
            double a = recurrent_alpha(alphas, index, 1.0);
            double b = recurrent_alpha(betas, index, 0.0);
            return a * x + b;
        }
        case 4: {
            double a = recurrent_alpha(alphas, index, 0.01);
            return x >= 0.0 ? x : a * x;
        }
        case 5: {
            double a = recurrent_alpha(alphas, index, 1.0);
            return x >= a ? x : 0.0;
        }
        case 6: {
            double a = recurrent_alpha(alphas, index, 1.0);
            double b = recurrent_alpha(betas, index, 1.0);
            return a * tanh(b * x);
        }
        case 7: {
            double a = recurrent_alpha(alphas, index, 0.2);
            double b = recurrent_alpha(betas, index, 0.5);
            double y = a * x + b;
            if (y < 0.0) return 0.0;
            if (y > 1.0) return 1.0;
            return y;
        }
        case 8: {
            double a = recurrent_alpha(alphas, index, 1.0);
            return x >= 0.0 ? x : a * (exp(x) - 1.0);
        }
        case 9:
            return x / (1.0 + fabs(x));
        case 10:
            return log1p(exp(x));
        case 0:
        default:
            return tanh(x);
    }
}

// 实现 `recurrent_activation_code` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_activation_code(const int* activations, int num_activations, int index, int default_code) {
    if (!activations || index >= num_activations) return default_code;
    return activations[index];
}

// 实现 `recurrent_num_dirs` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_num_dirs(int direction) {
    return direction == 2 ? 2 : 1;
}

// 实现 `recurrent_is_reverse` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_is_reverse(int direction, int dir_index) {
    return direction == 1 || (direction == 2 && dir_index == 1);
}

// 实现 `recurrent_x_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t recurrent_x_index(const Tensor* X, int layout, int t, int b, int i) {
    if (layout == 1) {
        int seq_len = X->shape[1];
        int input_size = X->shape[2];
        return ((size_t)b * seq_len * input_size) + ((size_t)t * input_size) + (size_t)i;
    }
    int batch = X->shape[1];
    int input_size = X->shape[2];
    return ((size_t)t * batch * input_size) + ((size_t)b * input_size) + (size_t)i;
}

// 实现 `recurrent_y_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t recurrent_y_index(const Tensor* Y, int layout, int t, int d, int b, int h) {
    if (layout == 1) {
        int seq_len = Y->shape[1];
        int num_dirs = Y->shape[2];
        int hidden = Y->shape[3];
        return ((size_t)b * seq_len * num_dirs * hidden)
             + ((size_t)t * num_dirs * hidden)
             + ((size_t)d * hidden)
             + (size_t)h;
    }
    int num_dirs = Y->shape[1];
    int batch = Y->shape[2];
    int hidden = Y->shape[3];
    return ((size_t)t * num_dirs * batch * hidden)
         + ((size_t)d * batch * hidden)
         + ((size_t)b * hidden)
         + (size_t)h;
}

// 实现 `recurrent_sequence_active` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_sequence_active(const Tensor* sequence_lens, int t, int b) {
    if (!sequence_lens || !sequence_lens->data) return 1;
    return get_value_as_int64(sequence_lens, (size_t)b) > t;
}

// 实现 `rnn` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `gru` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `lstm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `multinomial` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `loss_spatial_size` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t loss_spatial_size(const Tensor* input) {
    size_t spatial = 1;
    for (int i = 2; i < input->ndim; i++) spatial *= (size_t)input->shape[i];
    return spatial;
}

// 实现 `loss_target_weight` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double loss_target_weight(const Tensor* weight, int64_t cls) {
    if (!weight) return 1.0;
    return get_value_as_double(weight, (size_t)cls);
}

// 实现 `negative log likelihood loss` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `softmax cross entropy loss` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `nms_box_corners` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static void nms_box_corners(const Tensor* boxes, int batch, int box_idx, int center_point_box,
                            double* y1, double* x1, double* y2, double* x2) {
    int num_boxes = boxes->shape[1];
    size_t base = ((size_t)batch * num_boxes + (size_t)box_idx) * 4;
    double a = get_value_as_double(boxes, base + 0);
    double b = get_value_as_double(boxes, base + 1);
    double c = get_value_as_double(boxes, base + 2);
    double d = get_value_as_double(boxes, base + 3);

    if (center_point_box) {
        double x_center = a;
        double y_center = b;
        double width = c;
        double height = d;
        *y1 = y_center - height / 2.0;
        *x1 = x_center - width / 2.0;
        *y2 = y_center + height / 2.0;
        *x2 = x_center + width / 2.0;
    } else {
        *y1 = a;
        *x1 = b;
        *y2 = c;
        *x2 = d;
    }

    if (*y1 > *y2) {
        double tmp = *y1;
        *y1 = *y2;
        *y2 = tmp;
    }
    if (*x1 > *x2) {
        double tmp = *x1;
        *x1 = *x2;
        *x2 = tmp;
    }
}

// 实现 `nms_iou` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double nms_iou(const Tensor* boxes, int batch, int lhs, int rhs, int center_point_box) {
    double ay1, ax1, ay2, ax2;
    double by1, bx1, by2, bx2;
    nms_box_corners(boxes, batch, lhs, center_point_box, &ay1, &ax1, &ay2, &ax2);
    nms_box_corners(boxes, batch, rhs, center_point_box, &by1, &bx1, &by2, &bx2);

    double inter_h = fmax(0.0, fmin(ay2, by2) - fmax(ay1, by1));
    double inter_w = fmax(0.0, fmin(ax2, bx2) - fmax(ax1, bx1));
    double inter = inter_h * inter_w;
    double area_a = fmax(0.0, ay2 - ay1) * fmax(0.0, ax2 - ax1);
    double area_b = fmax(0.0, by2 - by1) * fmax(0.0, bx2 - bx1);
    double union_area = area_a + area_b - inter;
    return union_area <= 0.0 ? 0.0 : inter / union_area;
}

// 实现 `non max suppression` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `grid_denormalize` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_denormalize(double coord, int length, int align_corners) {
    if (align_corners) {
        return (coord + 1.0) * (double)(length - 1) / 2.0;
    }
    return ((coord + 1.0) * (double)length - 1.0) / 2.0;
}

// 实现 `grid_reflect_coordinate` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_reflect_coordinate(double coord, double low, double high) {
    if (high <= low) return low;
    double span = high - low;
    double value = fabs(fmod(coord - low, 2.0 * span));
    if (value > span) value = 2.0 * span - value;
    return value + low;
}

// 实现 `grid_sample_coordinate` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_sample_coordinate(double coord, int length, int padding_mode, int align_corners) {
    if (padding_mode == 1) {
        return fmin(fmax(coord, 0.0), (double)(length - 1));
    }
    if (padding_mode == 2) {
        double low = align_corners ? 0.0 : -0.5;
        double high = align_corners ? (double)(length - 1) : (double)length - 0.5;
        double reflected = grid_reflect_coordinate(coord, low, high);
        return fmin(fmax(reflected, 0.0), (double)(length - 1));
    }
    return coord;
}

// 实现 `grid_get_pixel_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_get_pixel_2d(const Tensor* input, int n, int c, double y, double x,
                                int padding_mode, int align_corners) {
    int height = input->shape[2];
    int width = input->shape[3];
    if (padding_mode == 1 || padding_mode == 2) {
        y = grid_sample_coordinate(y, height, padding_mode, align_corners);
        x = grid_sample_coordinate(x, width, padding_mode, align_corners);
    }
    int yi = (int)y;
    int xi = (int)x;
    if (yi < 0 || yi >= height || xi < 0 || xi >= width) return 0.0;
    size_t idx = ((size_t)n * input->shape[1] * height * width)
               + ((size_t)c * height * width)
               + ((size_t)yi * width)
               + (size_t)xi;
    return get_value_as_double(input, idx);
}

// 实现 `grid_bilinear_sample_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_bilinear_sample_2d(const Tensor* input, int n, int c, double y, double x,
                                      int padding_mode, int align_corners) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    double ly = y - (double)y0;
    double lx = x - (double)x0;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    return grid_get_pixel_2d(input, n, c, y0, x0, padding_mode, align_corners) * hy * hx
         + grid_get_pixel_2d(input, n, c, y0, x1, padding_mode, align_corners) * hy * lx
         + grid_get_pixel_2d(input, n, c, y1, x0, padding_mode, align_corners) * ly * hx
         + grid_get_pixel_2d(input, n, c, y1, x1, padding_mode, align_corners) * ly * lx;
}

// 实现 `grid_cubic_coefficients` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static void grid_cubic_coefficients(double t, double coeffs[4]) {
    double alpha = -0.75;
    double x = fabs(t);
    coeffs[0] = ((alpha * (x + 1.0) - 5.0 * alpha) * (x + 1.0) + 8.0 * alpha) * (x + 1.0) - 4.0 * alpha;
    coeffs[1] = ((alpha + 2.0) * x - (alpha + 3.0)) * x * x + 1.0;
    coeffs[2] = ((alpha + 2.0) * (1.0 - x) - (alpha + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    coeffs[3] = ((alpha * (2.0 - x) - 5.0 * alpha) * (2.0 - x) + 8.0 * alpha) * (2.0 - x) - 4.0 * alpha;
}

// 实现 `grid_bicubic_sample_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_bicubic_sample_2d(const Tensor* input, int n, int c, double y, double x,
                                     int padding_mode, int align_corners) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    double cy[4];
    double cx[4];
    grid_cubic_coefficients(y - (double)y0, cy);
    grid_cubic_coefficients(x - (double)x0, cx);
    double total = 0.0;
    for (int iy = 0; iy < 4; iy++) {
        for (int ix = 0; ix < 4; ix++) {
            total += cy[iy] * cx[ix] * grid_get_pixel_2d(
                input, n, c, y0 - 1 + iy, x0 - 1 + ix, padding_mode, align_corners
            );
        }
    }
    return total;
}

// 实现 `grid sample` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `lrn` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `mean variance normalization` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `eye like` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Clip：支持全广播
// 调用此函数前，Python 端已将 input, min_t, max_t 广播为相同形状
// 实现 `clip` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// MatMul 实现 (无加速)
// 实现 `matmul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `matmul integer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `qlinear matmul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Gather 实现
// 实现 `gather` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Expand 实现
// 实现 `expand` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Shape 实现
// 实现 `shape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 比较 A 和 B，结果存入 O (通常是 uint8)
// 展开 `BINARY_COMP_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define BINARY_COMP_IMPL(FUNC_NAME, COMPARE_OP) \
void FUNC_NAME(const Tensor* A, const Tensor* B, Tensor* O) { \
    if (!A || !B || !O) return; \
    size_t loop_size = O->size; \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < loop_size; i++) { \
        /* ONNX 规范：True 为 1, False 为 0 */ \
        uint8_t res = compare_tensor_values(A, i, B, i, COMPARE_OP) ? 1 : 0; \
        set_tensor_value_from_int(O, i, res); \
    } \
}


// Not: 按位取反 (bool/uint8) 或 逻辑非
// 实现 `not` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `isnan` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 输入已经被看作 boolean
// 展开 `BINARY_LOGIC_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define BINARY_LOGIC_IMPL(FUNC_NAME, OP_LOGIC) \
void FUNC_NAME(const Tensor* A, const Tensor* B, Tensor* O) { \
    if (!A || !B || !O) return; \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < O->size; i++) { \
        double val_a = get_value_as_double(A, i); \
        double val_b = get_value_as_double(B, i); \
        int bool_a = (val_a != 0); \
        int bool_b = (val_b != 0); \
        uint8_t res = (OP_LOGIC) ? 1 : 0; \
        set_tensor_value_from_int(O, i, res); \
    } \
}



// 实现 `sign` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `identity` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `mod` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `where` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ConstantOfShape
// 实现 `constant of shape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Range
// 实现 `range` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Tile
// 输入坐标 = 输出坐标 % 输入维度
// 实现 `tile` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Pad
// mode: 0=constant, 1=reflect, 2=edge
// 实现 `pad` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 检查某个轴是否在归约列表中
// 实现 `is_axis_reduced` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int is_axis_reduced(int axis, int* axes, int num_axes) {
    for (int i = 0; i < num_axes; i++) {
        if (axes[i] == axis) return 1;
    }
    return 0;
}

// 模板：通用归约内核
// 遍历输出的每个元素 (out_idx)。
// 根据 out_idx 反解出 "基准坐标" (base_coords)。
// 对于被归约的轴，基准坐标暂时设为 0；对于保留的轴，就是输出的对应坐标。
// 启动内层循环，遍历所有被归约维度的组合，更新 accumulator。
// 展开 `REDUCE_OP_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define REDUCE_OP_IMPL(FUNC_NAME, INIT_VAL, REDUCE_LOGIC, POST_PROC) \
void FUNC_NAME(const Tensor* input, Tensor* output, ReduceParams* params) { \
    if (!input || !output || !params) return; \
    int ndim = input->ndim; \
    int* axes = params->axes; \
    int num_axes = params->num_axes; \
    \
    /* 预计算归约的总步数 */ \
    size_t reduce_total_steps = 1; \
    for (int i = 0; i < num_axes; i++) { \
        reduce_total_steps *= input->shape[axes[i]]; \
    } \
    \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < output->size; i++) { \
        int coords[MAX_NDIM]; /* 当前处理的输入坐标 */ \
        int out_coords[MAX_NDIM]; /* 输出坐标 */ \
        \
        /* 反解输出坐标 */ \
        get_coords_from_index(i, out_coords, output->shape, output->ndim); \
        \
        /* 初始化输入坐标：保留维度填入 out_coords，归约维度填 0 */ \
        if (params->keepdims) { \
            for (int d = 0; d < ndim; d++) { \
                coords[d] = is_axis_reduced(d, axes, num_axes) ? 0 : out_coords[d]; \
            } \
        } else { \
            int out_dim_idx = 0; \
            for (int d = 0; d < ndim; d++) { \
                if (is_axis_reduced(d, axes, num_axes)) { \
                    coords[d] = 0; /* 归约轴初始化为 0 */ \
                } else { \
                    coords[d] = out_coords[out_dim_idx++]; \
                } \
            } \
        } \
        \
        /* 初始化累加器 */ \
        double acc = INIT_VAL; \
        \
        /* 内层循环：遍历归约空间 */ \
        for (size_t r = 0; r < reduce_total_steps; r++) { \
            /* 动态更新归约轴的坐标 */ \
            size_t temp_r = r; \
            for (int k = num_axes - 1; k >= 0; k--) { \
                int axis_idx = axes[k]; \
                int dim_size = input->shape[axis_idx]; \
                coords[axis_idx] = temp_r % dim_size; \
                temp_r /= dim_size; \
            } \
            \
            /* 读取输入并归约 */ \
            size_t in_idx = get_index_from_coords(coords, input->shape, ndim); \
            double val = get_value_as_double(input, in_idx); \
            REDUCE_LOGIC; \
        } \
        \
        /* 后处理并写入 */ \
        POST_PROC; \
        set_tensor_value_from_float(output, i, acc); \
    } \
}

// ReduceSum: Init=0, Acc+=val
// ReduceMean: Init=0, Acc+=val, Post=acc/count
// ReduceProd: Init=1, Acc*=val
// ReduceMax: Init=-inf, Acc=max
// ReduceMin: Init=+inf, Acc=min

// 展开 `ARG_OP_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define ARG_OP_IMPL(FUNC_NAME, INIT_VAL, CMP_OP, COMPARE_OP) \
void FUNC_NAME(const Tensor* input, Tensor* output, int axis, int select_last_index) { \
    if (!input || !output) return; \
    int ndim = input->ndim; \
    int axis_dim = input->shape[axis]; \
    int keepdims = (output->ndim == input->ndim); \
    \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < output->size; i++) { \
        int coords[MAX_NDIM]; \
        int out_coords[MAX_NDIM]; \
        get_coords_from_index(i, out_coords, output->shape, output->ndim); \
        \
        /* 映射坐标：输出坐标 -> 输入坐标 (归约轴置0) */ \
        int out_ptr = 0; \
        for (int d = 0; d < ndim; d++) { \
            if (d == axis) coords[d] = 0; \
            else { \
                coords[d] = out_coords[out_ptr]; \
                out_ptr++; \
            } \
            if (keepdims && d == axis) out_ptr++; \
        } \
        \
        /* 搜索最值 */ \
        int64_t best_idx = 0; \
        if (is_integer_dtype(input->dtype)) { \
            coords[axis] = 0; \
            size_t best_input_idx = get_index_from_coords(coords, input->shape, ndim); \
            for (int k = 1; k < axis_dim; k++) { \
                coords[axis] = k; \
                size_t in_idx = get_index_from_coords(coords, input->shape, ndim); \
                int better = compare_tensor_values(input, in_idx, input, best_input_idx, COMPARE_OP); \
                int equal = compare_tensor_values(input, in_idx, input, best_input_idx, TENSOR_COMPARE_EQ); \
                if (better || (select_last_index && equal)) { \
                    best_input_idx = in_idx; \
                    best_idx = k; \
                } \
            } \
        } else { \
            double best_val = INIT_VAL; \
            for (int k = 0; k < axis_dim; k++) { \
                coords[axis] = k; \
                size_t in_idx = get_index_from_coords(coords, input->shape, ndim); \
                double val = get_value_as_double(input, in_idx); \
                \
                /* 根据 select_last_index 决定相等时的行为 */ \
                int update = 0; \
                if (select_last_index) { \
                    if (val CMP_OP best_val || val == best_val) update = 1; /* >= 或 <= */ \
                } else { \
                    if (val CMP_OP best_val) update = 1; /* > 或 < */ \
                } \
                \
                if (update) { \
                    best_val = val; \
                    best_idx = k; \
                } \
            } \
        } \
        set_tensor_value_from_int(output, i, best_idx); \
    } \
}

//ArgMax和ArgMin


// 展开 `OMP_ATOMIC_DISPATCH` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define OMP_ATOMIC_DISPATCH(DTYPE_ENUM, C_TYPE, OP) \
    case DTYPE_ENUM: { \
        C_TYPE* ptr = (C_TYPE*)data->data; \
        C_TYPE v = (C_TYPE)val; \
        _Pragma("omp atomic") \
        ptr[data_idx] OP v; \
        break; \
    }

// ScatterND
// 遍历 updates，将其值写入 data 的指定位置
// 实现 `scatter nd` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GatherND
// 遍历 output，根据 indices 构造 data 坐标读取数据
// 实现 `gather nd` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GatherElements
// 实现 `gather elements` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// NonZero
// 实现 `nonzero` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Resize
// 实现 `resize` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 降序比较函数
// 作为 `compare_desc` 排序比较函数，保证排序类算子的值和索引顺序稳定。

// 升序比较函数
// 作为 `compare_asc` 排序比较函数，保证排序类算子的值和索引顺序稳定。

// 实现 `topk` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `cumsum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `simple_lcg` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static uint32_t simple_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

// 实现 `random uniform like` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `einsum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 展开 `UNARY_OP_WITH_ALPHA_IMPL` 相关的重复 C 实现，保持多个算子入口与 ctypes ABI 的循环逻辑一致。
#define UNARY_OP_WITH_ALPHA_IMPL(FUNC_NAME, MATH_LOGIC) \
void FUNC_NAME(const Tensor* input, Tensor* output, float alpha) { \
    if (!input || !output) return; \
    double a = (double)alpha; \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < input->size; i++) { \
        double val = get_value_as_double(input, i); \
        double res = MATH_LOGIC; \
        set_tensor_value_from_float(output, i, res); \
    } \
}

// Elu: x > 0 ? x : alpha * (exp(x) - 1)

// LeakyRelu: x >= 0 ? x : alpha * x

// ThresholdedRelu: x > alpha ? x : 0

// Celu: x >= 0 ? x : alpha * (exp(x/alpha) - 1)

// Selu: gamma * (x > 0 ? x : alpha * (exp(x) - 1))
// 实现 `selu` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// HardSigmoid: max(0, min(1, alpha * x + beta))
// 实现 `hard sigmoid` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Softplus: ln(1 + exp(x))

// Softsign: x / (1 + |x|)

// HardSwish: x * max(0, min(1, alpha * x + beta)), default alpha=1/6, beta=0.5
// x * max(0, min(1, x/6 + 0.5))

// Shrink: x < -lambd ? x + bias : (x > lambd ? x - bias : 0)
// 实现 `shrink` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Acos: arccos(x)

// Asin: arcsin(x)

// Cosh: (exp(x) + exp(-x)) / 2

// Sinh: (exp(x) - exp(-x)) / 2

// Asinh: ln(x + sqrt(x^2 + 1))

// Acosh: ln(x + sqrt(x^2 - 1)), for x >= 1

// Atanh: 0.5 * ln((1+x)/(1-x)), for |x| < 1

// 位运算逻辑
// 实现 `op_bitwise_and` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_bitwise_and(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a & (uint64_t)b, 64); }
// 实现 `op_bitwise_and_u` 的无符号整数版本，保持底层位模式语义。
static inline uint64_t op_bitwise_and_u(uint64_t a, uint64_t b) { return a & b; }
// 实现 `op_bitwise_or` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_bitwise_or(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a | (uint64_t)b, 64); }
// 实现 `op_bitwise_or_u` 的无符号整数版本，保持底层位模式语义。
static inline uint64_t op_bitwise_or_u(uint64_t a, uint64_t b) { return a | b; }
// 实现 `op_bitwise_xor` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_bitwise_xor(int64_t a, int64_t b) { return sign_extend_integer_bits((uint64_t)a ^ (uint64_t)b, 64); }
// 实现 `op_bitwise_xor_u` 的无符号整数版本，保持底层位模式语义。
static inline uint64_t op_bitwise_xor_u(uint64_t a, uint64_t b) { return a ^ b; }
// 实现 `op_shift_left` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_shift_left(int64_t a, int64_t b) { return b < 0 || b >= 64 ? 0 : sign_extend_integer_bits((uint64_t)a << (uint64_t)b, 64); }
// 实现 `op_shift_left_u` 的无符号整数版本，避免大位移触发 C 未定义行为。
static inline uint64_t op_shift_left_u(uint64_t a, uint64_t b) { return b >= 64 ? 0 : (a << b); }
// 实现 `op_shift_right` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static inline int64_t op_shift_right(int64_t a, int64_t b) { return b < 0 || b >= 64 ? 0 : a >> b; }
// 实现 `op_shift_right_u` 的无符号整数版本，使用逻辑右移。
static inline uint64_t op_shift_right_u(uint64_t a, uint64_t b) { return b >= 64 ? 0 : (a >> b); }

// BitwiseAnd
// 实现 `bitwise and` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// BitwiseOr
// 实现 `bitwise or` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// BitwiseXor
// 实现 `bitwise xor` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// BitwiseNot
// 实现 `bitwise not` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// BitShift
// direction: 0=LEFT, 1=RIGHT
// 实现 `bit shift` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ReduceL1: Sum(|x|)

// ReduceL2: Sqrt(Sum(x^2))

// ReduceLogSum: Log(Sum(x))

// ReduceLogSumExp: Log(Sum(exp(x)))，仅实现基础定义

// ReduceSumSquare: Sum(x^2)

// AveragePool
// 实现 `average pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// LpPool
// 实现 `lp pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GlobalAveragePool
// 假设输入是 NCHW (或至少后两维是空间维度)，如果不符合则不执行
// 实现 `global average pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GlobalMaxPool
// 实现 `global max pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GlobalLpPool
// 实现 `global lp pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Mean (Element-wise)
// 实现 `mean` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `size` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// IsInf
// 实现 `isinf` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// OneHot
// indices: 输入索引
// values: [off_value, on_value] (2 element tensor)
// axis: 扩充的维度
// 实现 `one hot` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Tril / Triu
// 实现 `triangular` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ================== Group 7: Normalization & Math Extensions 实现 ==================

// Round: round to nearest integer

// Erf: error function

// BatchNormalization (Inference Mode)
// Y = (X - mean) / sqrt(var + eps) * scale + B
// 优化为: Y = X * A + K
// 其中 A = scale / sqrt(var + eps), K = B - mean * A
// 实现 `batch norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// InstanceNormalization
// 对每个 (n, c) 切片计算均值和方差，然后归一化
// 实现 `instance norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// LayerNormalization
// 沿着 axis 轴进行归一化 (通常 axis=-1)
// 实现 `layer norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 获取窗函数大小
// 封装 `get_window_size` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static int64_t get_window_size(const Tensor* size_tensor) {
    if (!size_tensor) return 0;
    return get_value_as_int64(size_tensor, 0);
}

// Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
// 实现 `hann window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Hamming Window: 0.54 - 0.46 * cos(2*pi*n / (N-1))
// 实现 `hamming window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Blackman Window: 0.42 - 0.5*cos(...) + 0.08*cos(...)
// 实现 `blackman window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// RandomNormal: Box-Muller 变换
// 实现 `random normal` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Bernoulli: 生成 0 或 1
// 实现 `bernoulli` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Dropout (Inference Mode)
// 实现 `dropout` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Gelu

// 实现 `mish` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Hardmax
// 实现 `hardmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// LogSoftmax: x - max - log(sum(exp(x - max)))
// 实现 `log softmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// LpNormalization
// y = x / ||x||_p
// 实现 `lp normalization` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// DepthToSpace
// 实现 `depth to space` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// SpaceToDepth
// 实现 `space to depth` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ReverseSequence
// 实现 `reverse sequence` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Compress
// 实现 `compress` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// ScatterElements
// 实现 `scatter elements` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// GroupNormalization
// 实现 `group norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// Binarizer
// 实现 `binarizer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// DynamicQuantizeLinear
// 仅支持映射到 uint8 ([0, 255])
// 实现 `dynamic quantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif /* TENSOR_OPS_INTERNAL_H */
