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

// Egor Izmaylov: Function `float_to_bits` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline uint32_t float_to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Egor Izmaylov: Function `bits_to_float` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline float bits_to_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// 获取数据类型的字节大小
// Egor Izmaylov: Function `get_dtype_size` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT8_E4M3:
        case DTYPE_FLOAT8_E5M2:
        case DTYPE_INT4:
        case DTYPE_INT8:
        case DTYPE_UINT8:
            return 1;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
        case DTYPE_INT16:
            return 2;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
            return 4;
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
            return 8;
        default:
            return 4;
    }
}

// 用于排序
typedef struct {
    double value;
    int64_t index;
} TopKElement;

// 4-bit 饱和截断
// Egor Izmaylov: Function `saturate_cast_int4` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline int8_t saturate_cast_int4(int64_t val) {
    if (val > 7) return 7;
    if (val < -8) return -8;
    return (int8_t)val;
}

// 8-bit 饱和截断
// Egor Izmaylov: Function `saturate_cast_int8` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline int8_t saturate_cast_int8(int64_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// 8-bit 无符号饱和截断 (0 ~ 255)
// Egor Izmaylov: Function `saturate_cast_uint8` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline uint8_t saturate_cast_uint8(int64_t val) {
    if (val > 255) return 255;
    if (val < 0) return 0;
    return (uint8_t)val;
}

// 16-bit 饱和截断
// Egor Izmaylov: Function `saturate_cast_int16` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline int16_t saturate_cast_int16(int64_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// 32-bit 饱和截断
// Egor Izmaylov: Function `saturate_cast_int32` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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
// Egor Izmaylov: Function `float_to_float16` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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
// Egor Izmaylov: Function `float16_to_float` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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
// Egor Izmaylov: Function `float_to_bfloat16` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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
// Egor Izmaylov: Function `bfloat16_to_float` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
static inline float bfloat16_to_float(uint16_t value) {
    // 提取符号位
    uint32_t sign = (value & 0x8000) << 16;
    // 提取指数位
    uint32_t exp = (value & 0x7F80) << 16;
    // 提取尾数位
    uint32_t frac = (value & 0x007F) << 16;
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
// Egor Izmaylov: Function `fp8_e4m3_to_float` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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

// Egor Izmaylov: Function `float_to_fp8_e4m3` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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
// Egor Izmaylov: Function `fp8_e5m2_to_float` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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

// Egor Izmaylov: Function `float_to_fp8_e5m2` implements numeric conversion or saturation rules, keeping low-precision storage compatible with ONNX dtype expectations.
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

/**
 * 创建张量
 * 
 * @param shape 张量形状数组
 * @param ndim 张量维度数
 * @param dtype 数据类型
 * @return 创建的张量指针
 */
// Egor Izmaylov: Function `create_tensor` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
Tensor* create_tensor(int* shape, int ndim, DataType dtype) {
    if (ndim < 0) {
        return NULL;
    }

    // 分配张量结构体内存
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        return NULL;
    }
    
    // 设置维度数
    tensor->ndim = ndim;
    
    // 分配并复制形状数组
    tensor->shape = NULL;
    if (ndim > 0) {
        if (!shape) {
            free(tensor);
            return NULL;
        }
        tensor->shape = (int*)malloc(ndim * sizeof(int));
        if (!tensor->shape) {
            free(tensor);
            return NULL;
        }
        memcpy(tensor->shape, shape, ndim * sizeof(int));
    }
    
    // 设置数据类型
    tensor->dtype = dtype;
    
    // 计算总元素数
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
        tensor->size *= shape[i];
    }
    
    // 根据数据类型分配数据内存
    size_t elem_size = 0;
    switch (dtype) {
        case DTYPE_FLOAT8_E4M3:
        case DTYPE_FLOAT8_E5M2:
            elem_size = 1;  // 8位浮点数
            break;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            elem_size = 2;  // 16位浮点数
            break;
        case DTYPE_FLOAT32:
            elem_size = 4;  // 32位浮点数
            break;
        case DTYPE_FLOAT64:
            elem_size = 8;  // 64位浮点数
            break;
        case DTYPE_INT4:
            elem_size = 1;  // 4位整数
            break;
        case DTYPE_INT8:
            elem_size = 1;  // 8位整数
            break;
        case DTYPE_UINT8:
            elem_size = 1;  // 8位无符号整数
            break;
        case DTYPE_INT16:
            elem_size = 2;  // 16位整数
            break;
        case DTYPE_INT32:
            elem_size = 4;  // 32位整数
            break;
        case DTYPE_INT64:
            elem_size = 8;  // 64位整数
            break;
        default:
            elem_size = 4;  // 默认32位
            break;
    }
    
    // 分配数据内存
    //tensor->data = malloc(tensor->size * elem_size);
    size_t alloc_count = tensor->size == 0 ? 1 : tensor->size;
    tensor->data = calloc(alloc_count, elem_size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    return tensor;
}

/**
 * 释放张量内存
 * 
 * @param tensor 要释放的张量指针
 */
// Egor Izmaylov: Function `free_tensor` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
void free_tensor(Tensor* tensor) {
    // 检查张量指针是否有效
    if (tensor) {
        // 释放数据内存
        free(tensor->data);
        // 释放形状数组内存
        free(tensor->shape);
        // 释放张量结构体内存
        free(tensor);
    }
}

/*
 *
 * 从张量获取值，并作为 float 返回
 */
// Egor Izmaylov: Function `get_value_as_float` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline float get_value_as_float(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: return fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2: return fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]);
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
        case DTYPE_INT8: return (float)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (float)((uint8_t*)tensor->data)[index];
        case DTYPE_INT16: return (float)((int16_t*)tensor->data)[index];
        case DTYPE_INT32: return (float)((int32_t*)tensor->data)[index];
        case DTYPE_INT64: return (float)((int64_t*)tensor->data)[index];
        default: return 0.0f;
    }
}

/*
 *
 * 从张量获取值，并作为 double 返回
 */
// Egor Izmaylov: Function `get_value_as_double` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline double get_value_as_double(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: return (double)fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]);
        case DTYPE_FLOAT8_E5M2: return (double)fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]);
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
        case DTYPE_INT8: return (double)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (double)((uint8_t*)tensor->data)[index];
        case DTYPE_INT16: return (double)((int16_t*)tensor->data)[index];
        case DTYPE_INT32: return (double)((int32_t*)tensor->data)[index];
        case DTYPE_INT64: return (double)((int64_t*)tensor->data)[index];
        case DTYPE_FLOAT64: return ((double*)tensor->data)[index];
        default: return 0.0;
    }
}

/*
 *
 * 从张量获取值，并作为 int64_t 返回
 */
// Egor Izmaylov: Function `get_value_as_int64` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline int64_t get_value_as_int64(const Tensor* tensor, size_t index) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT32: return (int64_t)rintf(((float*)tensor->data)[index]);
        case DTYPE_FLOAT16: return (int64_t)rintf(float16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_BFLOAT16: return (int64_t)rintf(bfloat16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E4M3: return (int64_t)rintf(fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E5M2: return (int64_t)rintf(fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]));
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
        case DTYPE_INT8: return (int64_t)((int8_t*)tensor->data)[index];
        case DTYPE_UINT8: return (int64_t)((uint8_t*)tensor->data)[index];
        case DTYPE_INT16: return (int64_t)((int16_t*)tensor->data)[index];
        case DTYPE_INT32: return (int64_t)((int32_t*)tensor->data)[index];
        case DTYPE_INT64: return ((int64_t*)tensor->data)[index];
        case DTYPE_FLOAT64: return (int64_t)rint(((double*)tensor->data)[index]);
        default: return 0;
    }
}

/* 
 * 通用写入函数
 * 负责将计算结果安全地写入输出张量
 */
// Egor Izmaylov: Function `set_tensor_value_from_int` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline void set_tensor_value_from_int(Tensor* tensor, size_t index, int64_t value) {
    switch (tensor->dtype) {
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = saturate_cast_int4(value); break;
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = saturate_cast_int8(value); break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = saturate_cast_uint8(value); break;
        case DTYPE_INT16:   ((int16_t*)tensor->data)[index] = saturate_cast_int16(value); break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = saturate_cast_int32(value); break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = value; break;
        // 如果目标是浮点，进行转换
        case DTYPE_FLOAT8_E4M3: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3((float)value); break;
        case DTYPE_FLOAT8_E5M2: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2((float)value); break;
        case DTYPE_FLOAT16:     ((uint16_t*)tensor->data)[index] = float_to_float16((float)value); break;
        case DTYPE_BFLOAT16:    ((uint16_t*)tensor->data)[index] = float_to_bfloat16((float)value); break;
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = (double)value; break;
        default: break;
    }
}

// Egor Izmaylov: Function `set_tensor_value_from_float` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline void set_tensor_value_from_float(Tensor* tensor, size_t index, double value) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3((float)value); break;
        case DTYPE_FLOAT8_E5M2: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2((float)value); break;
        case DTYPE_FLOAT16:  ((uint16_t*)tensor->data)[index] = float_to_float16((float)value); break;
        case DTYPE_BFLOAT16: ((uint16_t*)tensor->data)[index] = float_to_bfloat16((float)value); break;
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = value; break;
        // 如果目标是整数，使用饱和截断转换
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = saturate_cast_int4((int64_t)rint(value)); break; 
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = saturate_cast_int8((int64_t)rint(value)); break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = saturate_cast_uint8((int64_t)rint(value)); break;
        case DTYPE_INT16:   ((int16_t*)tensor->data)[index] = saturate_cast_int16((int64_t)rint(value)); break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = saturate_cast_int32((int64_t)rint(value)); break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = (int64_t)rint(value); break;
        default: break;
    }
}

// Egor Izmaylov: Function `copy_tensor_element` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
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

/* 判断是否为整数类型 */
#define IS_INT_TYPE(d) (d == DTYPE_INT8 || d == DTYPE_UINT8 || d == DTYPE_INT16 || d == DTYPE_INT32 || d == DTYPE_INT64 || d == DTYPE_INT4)

// --- 通用一元算子宏模板 ---
#ifndef UNARY_OP_IMPL
// Egor Izmaylov: Macro `UNARY_OP_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
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
// Egor Izmaylov: Macro `BINARY_OP_INT_LOGIC` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
#define BINARY_OP_INT_LOGIC(OP_FUNC) \
    switch (O->dtype) { \
        case DTYPE_INT32: { \
            int32_t* out_data = (int32_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                out_data[i] = saturate_cast_int32(res); \
            } \
            break; \
        } \
        case DTYPE_INT16: { \
            int16_t* out_data = (int16_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                out_data[i] = saturate_cast_int16(res); \
            } \
            break; \
        } \
        case DTYPE_INT8: { \
            int8_t* out_data = (int8_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                out_data[i] = saturate_cast_int8(res); \
            } \
            break; \
        } \
        case DTYPE_UINT8: { \
            uint8_t* out_data = (uint8_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                out_data[i] = saturate_cast_uint8(res); \
            } \
            break; \
        } \
        case DTYPE_INT4: { \
            int8_t* out_data = (int8_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                int64_t res = OP_FUNC(val_a, val_b); \
                out_data[i] = saturate_cast_int4(res); \
            } \
            break; \
        } \
        case DTYPE_INT64: { \
            int64_t* out_data = (int64_t*)O->data; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < O->size; i++) { \
                int64_t val_a = get_value_as_int64(A, i); \
                int64_t val_b = get_value_as_int64(B, i); \
                out_data[i] = OP_FUNC(val_a, val_b); \
            } \
            break; \
        } \
        default: break; \
    }

// 简单的运算包装器，用于宏
// Egor Izmaylov: Function `op_add` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_add(int64_t a, int64_t b) { return a + b; }
// Egor Izmaylov: Function `op_sub` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_sub(int64_t a, int64_t b) { return a - b; }
// Egor Izmaylov: Function `op_mul` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_mul(int64_t a, int64_t b) { return a * b; }
// Egor Izmaylov: Function `op_div` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_div(int64_t a, int64_t b) { return b == 0 ? (a >= 0 ? INT64_MAX : INT64_MIN) : a / b; }

// 安全获取4D张量的值
// Egor Izmaylov: Function `get_val_4d_with_padding` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
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
// Egor Izmaylov: Function `relu_forward` is the C backend entry point for the relu operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `abs_forward` is the C backend entry point for the abs operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
            // 处理int64_min的特殊情况
            int64_t res = (val == INT64_MIN) ? INT64_MAX : (val < 0 ? -val : val);
            set_tensor_value_from_int(output, i, res);
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
// Egor Izmaylov: Function `init_cos_lut` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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
 * 使用查找表计算余弦值
 * 
 * @param x 输入角度（弧度）
 * @return 余弦值
 */
// Egor Izmaylov: Function `cos_lut_lookup` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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
// Egor Izmaylov: Function `cos_forward` is the C backend entry point for the cos operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `add_forward` is the C backend entry point for the add operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `sub_forward` is the C backend entry point for the sub operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `mul_forward` is the C backend entry point for the mul operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `div_forward` is the C backend entry point for the div operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `quantize_linear_forward` is the C backend entry point for the quantize linear operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void quantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y) {
    if (!X || !Scale || !ZeroPoint || !Y) return;
    
    size_t loop_size = Y->size;

    #pragma omp parallel for
    for (size_t i = 0; i < loop_size; i++) {
        double x_val = get_value_as_double(X, i);
        double s_val = get_value_as_double(Scale, i);
        double zp_val = get_value_as_double(ZeroPoint, i);
        
        double res = zp_val; 
        if (s_val != 0.0) {
            res = rint(x_val / s_val) + zp_val;
        }
        set_tensor_value_from_float(Y, i, res);
    }
}

// Egor Izmaylov: Function `dequantize_linear_forward` is the C backend entry point for the dequantize linear operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `conv2d_forward` is the C backend entry point for the conv2d operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void conv2d_forward(const Tensor* X, const Tensor* W, const Tensor* B, Tensor* Y, ConvParams* params) {
    // 形状解析
    // X: [Batch, InChannel, InH, InW]
    int batch = X->shape[0];
    int in_c  = X->shape[1];
    
    // W: [OutChannel, InChannel/Group, KernelH, KernelW]
    int out_c = W->shape[0];
    int k_h   = W->shape[2];
    int k_w   = W->shape[3];
    
    // Y: [Batch, OutChannel, OutH, OutW]
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    // 参数解析
    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    
    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;

    // 核心计算循环
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; n++) {
        for (int m = 0; m < out_c; m++) {
            // 当前 filter 属于第 g 个组
            int g = m / out_c_per_group;
            
            // 获取 Bias
            double bias_val = 0.0;
            if (B != NULL && B->data != NULL) {
                bias_val = get_value_as_double(B, m);
            }

            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double sum = 0.0;
                    // 卷积累加：在 Group 内遍历输入通道
                    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
                        // 实际的输入通道索引
                        int ic = g * in_c_per_group + ic_g;
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                // 计算输入特征图上的坐标 (包含 Dilation 和 Padding)
                                int h_in = oh * stride_h + kh * dilation_h - pad_top;
                                int w_in = ow * stride_w + kw * dilation_w - pad_left;
                                
                                // 获取输入值 (越界返回 0.0)
                                double val_x = get_val_4d_with_padding(X, n, ic, h_in, w_in, 0.0);
                                
                                // 获取权重值
                                // W 索引: m(out_c), ic_g(in_c_per_group), kh, kw
                                size_t w_idx = ((size_t)m * in_c_per_group * k_h * k_w) + 
                                               ((size_t)ic_g * k_h * k_w) + 
                                               ((size_t)kh * k_w) + kw;
                                double val_w = get_value_as_double(W, w_idx);
                                
                                sum += val_x * val_w;
                            }
                        }
                    }
                    
                    // 加上 Bias 并写入输出
                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) + 
                                   ((size_t)m * out_h * out_w) + 
                                   ((size_t)oh * out_w) + ow;
                    
                    set_tensor_value_from_float(Y, y_idx, sum + bias_val);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `conv_transpose2d_forward` is the C backend entry point for the conv transpose2d operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void conv_transpose2d_forward(const Tensor* X, const Tensor* W, const Tensor* B, Tensor* Y, ConvParams* params) {
    if (!X || !W || !Y || !params || X->ndim != 4 || W->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c = X->shape[1];
    int in_h = X->shape[2];
    int in_w = X->shape[3];
    int m_per_group = W->shape[1];
    int k_h = W->shape[2];
    int k_w = W->shape[3];
    int out_c = Y->shape[1];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    if (group <= 0 || stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0) return;
    if (in_c % group != 0 || out_c != m_per_group * group || W->shape[0] != in_c) return;

    int in_c_per_group = in_c / group;

    _Pragma("omp parallel for collapse(2)")
    for (int n = 0; n < batch; n++) {
        for (int oc = 0; oc < out_c; oc++) {
            int group_idx = oc / m_per_group;
            int oc_local = oc - group_idx * m_per_group;
            int ic_begin = group_idx * in_c_per_group;
            int ic_end = ic_begin + in_c_per_group;
            double bias_val = (B && B->data) ? get_value_as_double(B, oc) : 0.0;

            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double sum = bias_val;
                    for (int ic = ic_begin; ic < ic_end; ic++) {
                        for (int kh = 0; kh < k_h; kh++) {
                            int h_offset = oh + pad_top - kh * dilation_h;
                            if (h_offset % stride_h != 0) continue;
                            int ih = h_offset / stride_h;
                            if (ih < 0 || ih >= in_h) continue;

                            for (int kw = 0; kw < k_w; kw++) {
                                int w_offset = ow + pad_left - kw * dilation_w;
                                if (w_offset % stride_w != 0) continue;
                                int iw = w_offset / stride_w;
                                if (iw < 0 || iw >= in_w) continue;

                                size_t x_idx = ((size_t)n * in_c * in_h * in_w) +
                                               ((size_t)ic * in_h * in_w) +
                                               ((size_t)ih * in_w) + iw;
                                size_t w_idx = ((size_t)ic * m_per_group * k_h * k_w) +
                                               ((size_t)oc_local * k_h * k_w) +
                                               ((size_t)kh * k_w) + kw;
                                sum += get_value_as_double(X, x_idx) * get_value_as_double(W, w_idx);
                            }
                        }
                    }

                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)oc * out_h * out_w) +
                                   ((size_t)oh * out_w) + ow;
                    set_tensor_value_from_float(Y, y_idx, sum);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `conv_integer_forward` is the C backend entry point for the conv integer operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void conv_integer_forward(const Tensor* X, const Tensor* W,
                          const Tensor* XZeroPoint, const Tensor* WZeroPoint,
                          Tensor* Y, ConvParams* params) {
    if (!X || !W || !Y || !params || X->ndim != 4 || W->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c  = X->shape[1];
    int out_c = W->shape[0];
    int k_h   = W->shape[2];
    int k_w   = W->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    if (group <= 0) return;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;

    _Pragma("omp parallel for collapse(2)")
    for (int n = 0; n < batch; n++) {
        for (int m = 0; m < out_c; m++) {
            int g = m / out_c_per_group;
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int64_t sum = 0;
                    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
                        int ic = g * in_c_per_group + ic_g;
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                int h_in = oh * stride_h + kh * dilation_h - pad_top;
                                int w_in = ow * stride_w + kw * dilation_w - pad_left;
                                if (h_in < 0 || h_in >= X->shape[2] || w_in < 0 || w_in >= X->shape[3]) {
                                    continue;
                                }

                                size_t x_idx = ((size_t)n * in_c * X->shape[2] * X->shape[3]) +
                                               ((size_t)ic * X->shape[2] * X->shape[3]) +
                                               ((size_t)h_in * X->shape[3]) + w_in;
                                size_t w_idx = ((size_t)m * in_c_per_group * k_h * k_w) +
                                               ((size_t)ic_g * k_h * k_w) +
                                               ((size_t)kh * k_w) + kw;

                                int64_t x_val = get_value_as_int64(X, x_idx);
                                int64_t w_val = get_value_as_int64(W, w_idx);
                                int64_t x_zp = (XZeroPoint && XZeroPoint->data) ? get_value_as_int64(XZeroPoint, x_idx) : 0;
                                int64_t w_zp = (WZeroPoint && WZeroPoint->data) ? get_value_as_int64(WZeroPoint, w_idx) : 0;
                                sum += (x_val - x_zp) * (w_val - w_zp);
                            }
                        }
                    }

                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)m * out_h * out_w) +
                                   ((size_t)oh * out_w) + ow;
                    if (Y->dtype == DTYPE_INT32) {
                        ((int32_t*)Y->data)[y_idx] = (int32_t)sum;
                    } else {
                        set_tensor_value_from_int(Y, y_idx, sum);
                    }
                }
            }
        }
    }
}

// Egor Izmaylov: Function `qlinear_conv_forward` is the C backend entry point for the qlinear conv operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void qlinear_conv_forward(const Tensor* X, const Tensor* XScale, const Tensor* XZeroPoint,
                          const Tensor* W, const Tensor* WScale, const Tensor* WZeroPoint,
                          const Tensor* YScale, const Tensor* YZeroPoint,
                          const Tensor* Bias, Tensor* Y, ConvParams* params) {
    if (!X || !XScale || !W || !WScale || !YScale || !YZeroPoint || !Y || !params) return;
    if (X->ndim != 4 || W->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c  = X->shape[1];
    int out_c = W->shape[0];
    int k_h   = W->shape[2];
    int k_w   = W->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    if (group <= 0 || in_c % group != 0 || out_c % group != 0) return;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;

    _Pragma("omp parallel for collapse(2)")
    for (int n = 0; n < batch; n++) {
        for (int m = 0; m < out_c; m++) {
            int g = m / out_c_per_group;
            size_t w_scale_idx = (size_t)m * in_c_per_group * k_h * k_w;
            double w_scale = get_value_as_double(WScale, w_scale_idx);
            int64_t bias = (Bias && Bias->data) ? get_value_as_int64(Bias, m) : 0;

            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int64_t acc = 0;
                    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
                        int ic = g * in_c_per_group + ic_g;
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                int h_in = oh * stride_h + kh * dilation_h - pad_top;
                                int w_in = ow * stride_w + kw * dilation_w - pad_left;
                                if (h_in < 0 || h_in >= X->shape[2] || w_in < 0 || w_in >= X->shape[3]) {
                                    continue;
                                }

                                size_t x_idx = ((size_t)n * in_c * X->shape[2] * X->shape[3]) +
                                               ((size_t)ic * X->shape[2] * X->shape[3]) +
                                               ((size_t)h_in * X->shape[3]) + w_in;
                                size_t w_idx = ((size_t)m * in_c_per_group * k_h * k_w) +
                                               ((size_t)ic_g * k_h * k_w) +
                                               ((size_t)kh * k_w) + kw;

                                int64_t x_val = get_value_as_int64(X, x_idx);
                                int64_t w_val = get_value_as_int64(W, w_idx);
                                int64_t x_zp = (XZeroPoint && XZeroPoint->data) ? get_value_as_int64(XZeroPoint, x_idx) : 0;
                                int64_t w_zp = (WZeroPoint && WZeroPoint->data) ? get_value_as_int64(WZeroPoint, w_idx) : 0;
                                acc += (x_val - x_zp) * (w_val - w_zp);
                            }
                        }
                    }

                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)m * out_h * out_w) +
                                   ((size_t)oh * out_w) + ow;
                    double x_scale = get_value_as_double(XScale, 0);
                    double y_scale = get_value_as_double(YScale, y_idx);
                    double y_zp = get_value_as_double(YZeroPoint, y_idx);
                    double q = ((double)(acc + bias) * x_scale * w_scale) / y_scale + y_zp;
                    set_tensor_value_from_float(Y, y_idx, q);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `max_pool_forward` is the C backend entry point for the max pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void max_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params) {
    int batch = X->shape[0];
    int channels = X->shape[1];
    int in_h = X->shape[2];
    int in_w = X->shape[3];
    
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];
    
    int k_h = params->kernel_shape[0];
    int k_w = params->kernel_shape[1];
    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double max_val = -INFINITY; 
                    // 遍历 Kernel
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int h_in = oh * stride_h + kh * dilation_h - pad_top;
                            int w_in = ow * stride_w + kw * dilation_w - pad_left;
                            // MaxPool padding 策略: 只处理边界内
                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                                size_t x_idx = ((size_t)n * channels * in_h * in_w) + 
                                               ((size_t)c * in_h * in_w) + 
                                               ((size_t)h_in * in_w) + w_in;
                                double val = get_value_as_double(X, x_idx);
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    size_t y_idx = ((size_t)n * channels * out_h * out_w) + 
                                   ((size_t)c * out_h * out_w) + 
                                   ((size_t)oh * out_w) + ow;
                    set_tensor_value_from_float(Y, y_idx, max_val);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `max_unpool_forward` is the C backend entry point for the max unpool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void max_unpool_forward(const Tensor* X, const Tensor* Indices, Tensor* Y, PoolParams* params) {
    if (!X || !Indices || !Y || !params || !X->data || !Indices->data || !Y->data) return;
    if (X->ndim != Indices->ndim || X->ndim != Y->ndim || X->ndim < 3 || X->ndim > MAX_NDIM) return;
    if (X->size != Indices->size) return;

    int spatial_rank = X->ndim - 2;
    int inferred_shape[MAX_NDIM];
    inferred_shape[0] = X->shape[0];
    inferred_shape[1] = X->shape[1];
    for (int dim = 0; dim < spatial_rank; dim++) {
        int inferred = (X->shape[dim + 2] - 1) * params->strides[dim]
                       - params->pads[dim]
                       - params->pads[spatial_rank + dim]
                       + params->kernel_shape[dim];
        if (inferred <= 0) return;
        inferred_shape[dim + 2] = inferred;
    }

    int64_t inferred_total = 1;
    for (int dim = 0; dim < X->ndim; dim++) {
        inferred_total *= inferred_shape[dim];
    }

    for (size_t src_idx = 0; src_idx < X->size; src_idx++) {
        int64_t flat_index = get_value_as_int64(Indices, src_idx);
        if (flat_index < 0 || flat_index >= inferred_total) {
            continue;
        }

        int coords[MAX_NDIM];
        int64_t remaining = flat_index;
        for (int dim = X->ndim - 1; dim >= 0; dim--) {
            coords[dim] = (int)(remaining % inferred_shape[dim]);
            remaining /= inferred_shape[dim];
        }

        size_t dst_idx = 0;
        int in_bounds = 1;
        for (int dim = 0; dim < Y->ndim; dim++) {
            if (coords[dim] < 0 || coords[dim] >= Y->shape[dim]) {
                in_bounds = 0;
                break;
            }
            dst_idx = dst_idx * (size_t)Y->shape[dim] + (size_t)coords[dim];
        }
        if (in_bounds) {
            copy_tensor_element(Y, dst_idx, X, src_idx);
        }
    }
}

// Egor Izmaylov: Function `max_roi_pool_forward` is the C backend entry point for the max roi pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void max_roi_pool_forward(const Tensor* X, const Tensor* rois, Tensor* Y,
                          int pooled_h, int pooled_w, float spatial_scale) {
    if (!X || !rois || !Y || !X->data || !rois->data || !Y->data) return;
    if (X->ndim != 4 || rois->ndim != 2 || Y->ndim != 4 || rois->shape[1] != 5) return;
    if (pooled_h <= 0 || pooled_w <= 0) return;

    int num_rois = rois->shape[0];
    int batches = X->shape[0];
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (Y->shape[0] != num_rois || Y->shape[1] != channels ||
        Y->shape[2] != pooled_h || Y->shape[3] != pooled_w) return;

    _Pragma("omp parallel for collapse(4)")
    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < pooled_h; ph++) {
                for (int pw = 0; pw < pooled_w; pw++) {
                    size_t roi_base = (size_t)roi_idx * 5;
                    int batch = (int)get_value_as_int64(rois, roi_base);
                    if (batch < 0 || batch >= batches) continue;

                    int x1 = (int)nearbyint(get_value_as_double(rois, roi_base + 1) * (double)spatial_scale);
                    int y1 = (int)nearbyint(get_value_as_double(rois, roi_base + 2) * (double)spatial_scale);
                    int x2 = (int)nearbyint(get_value_as_double(rois, roi_base + 3) * (double)spatial_scale);
                    int y2 = (int)nearbyint(get_value_as_double(rois, roi_base + 4) * (double)spatial_scale);

                    int roi_w = x2 - x1 + 1;
                    int roi_h = y2 - y1 + 1;
                    if (roi_w < 1) roi_w = 1;
                    if (roi_h < 1) roi_h = 1;
                    double bin_h = (double)roi_h / (double)pooled_h;
                    double bin_w = (double)roi_w / (double)pooled_w;

                    int hstart = (int)floor((double)ph * bin_h) + y1;
                    int hend = (int)ceil((double)(ph + 1) * bin_h) + y1;
                    int wstart = (int)floor((double)pw * bin_w) + x1;
                    int wend = (int)ceil((double)(pw + 1) * bin_w) + x1;
                    if (hstart < 0) hstart = 0;
                    if (hend < 0) hend = 0;
                    if (wstart < 0) wstart = 0;
                    if (wend < 0) wend = 0;
                    if (hstart > height) hstart = height;
                    if (hend > height) hend = height;
                    if (wstart > width) wstart = width;
                    if (wend > width) wend = width;

                    double max_val = 0.0;
                    if (hend > hstart && wend > wstart) {
                        max_val = -DBL_MAX;
                        for (int h = hstart; h < hend; h++) {
                            for (int w = wstart; w < wend; w++) {
                                size_t x_idx = ((size_t)batch * channels * height * width)
                                             + ((size_t)c * height * width)
                                             + ((size_t)h * width)
                                             + (size_t)w;
                                double value = get_value_as_double(X, x_idx);
                                if (value > max_val) max_val = value;
                            }
                        }
                    }

                    size_t y_idx = ((size_t)roi_idx * channels * pooled_h * pooled_w)
                                 + ((size_t)c * pooled_h * pooled_w)
                                 + ((size_t)ph * pooled_w)
                                 + (size_t)pw;
                    set_tensor_value_from_float(Y, y_idx, max_val);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `roi_align_bilinear_sample` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `roi_align_max_weighted_term` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `roi_align_forward` is the C backend entry point for the roi align operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void roi_align_forward(const Tensor* X, const Tensor* rois, const Tensor* batch_indices, Tensor* Y,
                       int output_height, int output_width, int sampling_ratio,
                       float spatial_scale, int mode, int coordinate_transformation_mode) {
    if (!X || !rois || !batch_indices || !Y || !X->data || !rois->data || !batch_indices->data || !Y->data) return;
    if (X->ndim != 4 || rois->ndim != 2 || batch_indices->ndim != 1 || Y->ndim != 4 || rois->shape[1] != 4) return;
    if (output_height <= 0 || output_width <= 0) return;

    int num_rois = rois->shape[0];
    int batches = X->shape[0];
    int channels = X->shape[1];
    if (batch_indices->shape[0] != num_rois ||
        Y->shape[0] != num_rois || Y->shape[1] != channels ||
        Y->shape[2] != output_height || Y->shape[3] != output_width) return;

    int half_pixel = (coordinate_transformation_mode == 0);
    double offset = half_pixel ? 0.5 : 0.0;

    _Pragma("omp parallel for collapse(4)")
    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < output_height; ph++) {
                for (int pw = 0; pw < output_width; pw++) {
                    int batch = (int)get_value_as_int64(batch_indices, (size_t)roi_idx);
                    if (batch < 0 || batch >= batches) continue;

                    size_t roi_base = (size_t)roi_idx * 4;
                    double roi_start_w = get_value_as_double(rois, roi_base) * (double)spatial_scale - offset;
                    double roi_start_h = get_value_as_double(rois, roi_base + 1) * (double)spatial_scale - offset;
                    double roi_end_w = get_value_as_double(rois, roi_base + 2) * (double)spatial_scale - offset;
                    double roi_end_h = get_value_as_double(rois, roi_base + 3) * (double)spatial_scale - offset;
                    double roi_w = roi_end_w - roi_start_w;
                    double roi_h = roi_end_h - roi_start_h;
                    if (!half_pixel) {
                        if (roi_w < 1.0) roi_w = 1.0;
                        if (roi_h < 1.0) roi_h = 1.0;
                    }
                    double bin_h = roi_h / (double)output_height;
                    double bin_w = roi_w / (double)output_width;
                    int grid_h = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_h / (double)output_height);
                    int grid_w = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_w / (double)output_width);
                    if (grid_h < 1) grid_h = 1;
                    if (grid_w < 1) grid_w = 1;
                    int count = grid_h * grid_w;

                    double output_value = (mode == 1) ? -DBL_MAX : 0.0;
                    for (int iy = 0; iy < grid_h; iy++) {
                        double yy = roi_start_h + (double)ph * bin_h + ((double)iy + 0.5) * bin_h / (double)grid_h;
                        for (int ix = 0; ix < grid_w; ix++) {
                            double xx = roi_start_w + (double)pw * bin_w + ((double)ix + 0.5) * bin_w / (double)grid_w;
                            if (mode == 1) {
                                double term = roi_align_max_weighted_term(X, batch, c, yy, xx);
                                if (term > output_value) output_value = term;
                            } else {
                                output_value += roi_align_bilinear_sample(X, batch, c, yy, xx);
                            }
                        }
                    }
                    if (mode != 1) {
                        output_value /= (double)count;
                    }

                    size_t y_idx = ((size_t)roi_idx * channels * output_height * output_width)
                                 + ((size_t)c * output_height * output_width)
                                 + ((size_t)ph * output_width)
                                 + (size_t)pw;
                    set_tensor_value_from_float(Y, y_idx, output_value);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `gemm_forward` is the C backend entry point for the gemm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// ================== Softmax 实现 ==================
// Egor Izmaylov: Function `softmax_forward` is the C backend entry point for the softmax operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void softmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (axis < 0) axis += input->ndim;
    
    // 将 Tensor 视为 [Outer, Inner, Remaining]
    int inner_dim = input->shape[axis];
    
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) max_val = val;
            }
            double sum = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum += exp(val - max_val);
            }
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                double res = exp(val - max_val) / sum;
                set_tensor_value_from_float(output, idx, res);
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

// Flatten 实现
// Egor Izmaylov: Function `flatten_forward` is the C backend entry point for the flatten operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void flatten_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    size_t total_bytes = input->size * elem_size;
    memcpy(output->data, input->data, total_bytes);
}

// Reshape 实现
// Egor Izmaylov: Function `reshape_forward` is the C backend entry point for the reshape operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void reshape_forward(const Tensor* input, Tensor* output) {
    flatten_forward(input, output);
}

// 从平坦索引反解 N 维坐标
// Egor Izmaylov: Function `get_coords_from_index` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static inline void get_coords_from_index(size_t index, int* coords, int* shape, int ndim) {
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = index % shape[i];
        index /= shape[i];
    }
}

// 从 N 维坐标计算平坦索引
// Egor Izmaylov: Function `get_index_from_coords` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
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
// Egor Izmaylov: Function `transpose_forward` is the C backend entry point for the transpose operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void transpose_forward(const Tensor* input, Tensor* output, int* perm) {
    if (!input || !output || !perm) return;
    int ndim = input->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0}; // 输出坐标
        int in_coords[MAX_NDIM] = {0};  // 输入坐标
        
        // 1. 根据输出的平坦索引 i，反解出输出坐标
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        // 2. 映射回输入坐标
        // 规则：output[d] 对应 input[perm[d]]
        for (int k = 0; k < ndim; k++) {
            in_coords[perm[k]] = out_coords[k];
        }
        
        // 3. 计算输入的平坦索引
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        
        // 4. 搬运数据
        double val = get_value_as_double(input, in_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// 整数辅助函数
// Egor Izmaylov: Function `op_max` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_max(int64_t a, int64_t b) { return a > b ? a : b; }
// Egor Izmaylov: Function `op_min` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_min(int64_t a, int64_t b) { return a < b ? a : b; }

// Pow 实现
// Egor Izmaylov: Function `pow_forward` is the C backend entry point for the pow operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void pow_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double val_a = get_value_as_double(A, i);
        double val_b = get_value_as_double(B, i);
        double res = pow(val_a, val_b);
        set_tensor_value_from_float(O, i, res);
    }
}

// Max 实现
// Egor Izmaylov: Function `max_forward` is the C backend entry point for the max operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
            double res = (val_a > val_b ? val_a : val_b);
            set_tensor_value_from_float(O, i, res);
        }
    }
}

// Min 实现
// Egor Izmaylov: Function `min_forward` is the C backend entry point for the min operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
            double res = (val_a < val_b ? val_a : val_b);
            set_tensor_value_from_float(O, i, res);
        }
    }
}

// Egor Izmaylov: Function `concat_forward` is the C backend entry point for the concat operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void concat_forward(const Tensor** inputs, int num_inputs, Tensor* output, int axis) {
    if (!inputs || !output || num_inputs < 1) return;

    // 处理负轴
    int ndim = output->ndim;
    if (ndim > MAX_NDIM) {

        return;
    }
    
    // 缓存每个输入在 axis 维度的长度
    int input_dims[128]; // 假设输入数量不超过 128
    if (num_inputs > 128) return; 
    for (int k = 0; k < num_inputs; k++) {
        input_dims[k] = inputs[k]->shape[axis];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM]; // 最大维度为 16
        
        // 1. 反解输出坐标
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 2. 确定当前坐标落在哪个输入张量中
        int target_val = coords[axis];
        int input_idx = -1;
        int local_axis_val = target_val;
        
        for (int k = 0; k < num_inputs; k++) {
            if (local_axis_val < input_dims[k]) {
                input_idx = k;
                break;
            }
            local_axis_val -= input_dims[k];
        }
        
        if (input_idx >= 0) {
            // 3. 修正为局部坐标
            coords[axis] = local_axis_val;
            
            // 4. 读取源数据并写入
            const Tensor* src = inputs[input_idx];
            size_t src_idx = get_index_from_coords(coords, src->shape, ndim);
            copy_tensor_element(output, i, src, src_idx);
        }
    }
}

// Egor Izmaylov: Function `slice_forward` is the C backend entry point for the slice operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void slice_forward(const Tensor* input, Tensor* output, int* starts, int* steps) {
    if (!input || !output || !starts || !steps) return;
    
    int ndim = input->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
        
        // 1. 获取输出坐标
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        // 2. 映射回输入坐标: in = start + out * step
        for (int d = 0; d < ndim; d++) {
            in_coords[d] = starts[d] + out_coords[d] * steps[d];
        }
        
        // 3. 读写数据
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        copy_tensor_element(output, i, input, in_idx);
    }
}

// Neg
UNARY_OP_IMPL(neg_forward, -val)

// Reciprocal
UNARY_OP_IMPL(reciprocal_forward, 1.0 / val)

// Ceil
UNARY_OP_IMPL(ceil_forward, ceil(val))

// Floor
UNARY_OP_IMPL(floor_forward, floor(val))

// Cast
// 读取时自动转 double，写入 set_tensor_value 时会自动转为 output->dtype
// Egor Izmaylov: Function `cast_forward` is the C backend entry point for the cast operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void cast_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) return;

    if (input->dtype == output->dtype) {
        size_t elem_size = get_dtype_size(input->dtype);
        memcpy(output->data, input->data, input->size * elem_size);
        return;
    }
    
    // 检查是否是 "浮点 -> 整数" 的情况
    int is_float_to_int = (input->dtype == DTYPE_FLOAT32 || input->dtype == DTYPE_FLOAT64 || 
                           input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16) &&
                          IS_INT_TYPE(output->dtype);
    int is_int_to_int = IS_INT_TYPE(input->dtype) && IS_INT_TYPE(output->dtype);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        // 1. 读取输入 (统一转 double)
        double val = get_value_as_double(input, i);
        
        // 2. 写入输出
        if (is_float_to_int || is_int_to_int) {
            int64_t int_val = is_int_to_int ? get_value_as_int64(input, i) : (int64_t)val;
            set_tensor_value_from_int(output, i, int_val);
        } else {
            // 其他情况 (Int->Float, Float->Float, Int->Int) 保持原有逻辑
            set_tensor_value_from_float(output, i, val);
        }
    }
}

// Egor Izmaylov: Function `sum_forward` is the C backend entry point for the sum operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `prelu_forward` is the C backend entry point for the prelu operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `det_forward` is the C backend entry point for the det operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void det_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->ndim < 2) return;

    int n = input->shape[input->ndim - 1];
    int m = input->shape[input->ndim - 2];
    if (n != m || n <= 0) return;

    size_t matrix_size = (size_t)n * (size_t)n;
    size_t batch = output->size;

    _Pragma("omp parallel for")
    for (size_t b = 0; b < batch; b++) {
        double* work = (double*)malloc(matrix_size * sizeof(double));
        if (!work) continue;

        size_t base = b * matrix_size;
        for (size_t i = 0; i < matrix_size; i++) {
            work[i] = get_value_as_double(input, base + i);
        }

        double det = 1.0;
        int sign = 1;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double pivot_abs = fabs(work[(size_t)col * n + col]);
            for (int row = col + 1; row < n; row++) {
                double candidate = fabs(work[(size_t)row * n + col]);
                if (candidate > pivot_abs) {
                    pivot_abs = candidate;
                    pivot = row;
                }
            }

            if (pivot_abs == 0.0) {
                det = 0.0;
                break;
            }

            if (pivot != col) {
                for (int j = 0; j < n; j++) {
                    double tmp = work[(size_t)col * n + j];
                    work[(size_t)col * n + j] = work[(size_t)pivot * n + j];
                    work[(size_t)pivot * n + j] = tmp;
                }
                sign = -sign;
            }

            double pivot_val = work[(size_t)col * n + col];
            det *= pivot_val;
            for (int row = col + 1; row < n; row++) {
                double factor = work[(size_t)row * n + col] / pivot_val;
                work[(size_t)row * n + col] = 0.0;
                for (int j = col + 1; j < n; j++) {
                    work[(size_t)row * n + j] -= factor * work[(size_t)col * n + j];
                }
            }
        }

        set_tensor_value_from_float(output, b, det * sign);
        free(work);
    }
}

// Egor Izmaylov: Function `tensor_scalar_equal` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `tensor_scalar_compare` is a qsort comparator used by ranking-style operators, preserving deterministic ordering for values and original indices.
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

// Egor Izmaylov: Function `unique_forward` is the C backend entry point for the unique operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
int unique_forward(const Tensor* input, Tensor* values, Tensor* indices, Tensor* inverse, Tensor* counts, int sorted) {
    if (!input || !values || !indices || !inverse || !counts) return 0;
    if (!input->data || !values->data || !indices->data || !inverse->data || !counts->data) return 0;
    if (values->size < input->size || indices->size < input->size || inverse->size < input->size || counts->size < input->size) return 0;

    size_t n = input->size;
    size_t elem_size = get_dtype_size(values->dtype);
    size_t* first_indices = (size_t*)malloc((n == 0 ? 1 : n) * sizeof(size_t));
    int* order = (int*)malloc((n == 0 ? 1 : n) * sizeof(int));
    int* remap = (int*)malloc((n == 0 ? 1 : n) * sizeof(int));
    if (!first_indices || !order || !remap) {
        free(first_indices);
        free(order);
        free(remap);
        return 0;
    }

    int unique_count = 0;
    for (size_t i = 0; i < n; i++) {
        int found = -1;
        for (int j = 0; j < unique_count; j++) {
            if (tensor_scalar_equal(input, i, first_indices[j])) {
                found = j;
                break;
            }
        }

        if (found < 0) {
            first_indices[unique_count] = i;
            copy_tensor_element(values, unique_count, input, i);
            set_tensor_value_from_int(indices, unique_count, (int64_t)i);
            set_tensor_value_from_int(counts, unique_count, 1);
            set_tensor_value_from_int(inverse, i, unique_count);
            unique_count++;
        } else {
            int64_t old_count = get_value_as_int64(counts, found);
            set_tensor_value_from_int(counts, found, old_count + 1);
            set_tensor_value_from_int(inverse, i, found);
        }
    }

    if (sorted && unique_count > 1) {
        for (int i = 0; i < unique_count; i++) order[i] = i;
        for (int i = 1; i < unique_count; i++) {
            int current = order[i];
            int j = i - 1;
            while (j >= 0 && tensor_scalar_compare(input, first_indices[order[j]], first_indices[current]) > 0) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = current;
        }

        void* tmp_values = malloc((size_t)unique_count * elem_size);
        int64_t* tmp_indices = (int64_t*)malloc((size_t)unique_count * sizeof(int64_t));
        int64_t* tmp_counts = (int64_t*)malloc((size_t)unique_count * sizeof(int64_t));
        if (tmp_values && tmp_indices && tmp_counts) {
            for (int new_pos = 0; new_pos < unique_count; new_pos++) {
                int old_pos = order[new_pos];
                remap[old_pos] = new_pos;
                memcpy((uint8_t*)tmp_values + (size_t)new_pos * elem_size,
                       (uint8_t*)values->data + (size_t)old_pos * elem_size,
                       elem_size);
                tmp_indices[new_pos] = get_value_as_int64(indices, old_pos);
                tmp_counts[new_pos] = get_value_as_int64(counts, old_pos);
            }

            memcpy(values->data, tmp_values, (size_t)unique_count * elem_size);
            for (int i = 0; i < unique_count; i++) {
                set_tensor_value_from_int(indices, i, tmp_indices[i]);
                set_tensor_value_from_int(counts, i, tmp_counts[i]);
            }
            for (size_t i = 0; i < n; i++) {
                int old_inverse = (int)get_value_as_int64(inverse, i);
                set_tensor_value_from_int(inverse, i, remap[old_inverse]);
            }
        }
        free(tmp_values);
        free(tmp_indices);
        free(tmp_counts);
    }

    free(first_indices);
    free(order);
    free(remap);
    return unique_count;
}

// Egor Izmaylov: Function `hz_to_mel` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double hz_to_mel(double frequency) {
    return 2595.0 * log10(1.0 + frequency / 700.0);
}

// Egor Izmaylov: Function `mel_to_hz` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Egor Izmaylov: Function `mel_weight_matrix_forward` is the C backend entry point for the mel weight matrix operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void mel_weight_matrix_forward(const Tensor* num_mel_bins, const Tensor* dft_length,
                               const Tensor* sample_rate, const Tensor* lower_edge_hertz,
                               const Tensor* upper_edge_hertz, Tensor* output) {
    if (!num_mel_bins || !dft_length || !sample_rate || !lower_edge_hertz || !upper_edge_hertz || !output) return;

    int bins = (int)get_value_as_int64(num_mel_bins, 0);
    int dft_len = (int)get_value_as_int64(dft_length, 0);
    int rate = (int)get_value_as_int64(sample_rate, 0);
    double lower = get_value_as_double(lower_edge_hertz, 0);
    double upper = get_value_as_double(upper_edge_hertz, 0);
    if (bins < 0 || dft_len < 0 || rate <= 0 || upper < lower) return;

    int spectrogram_bins = dft_len / 2 + 1;
    if (output->ndim != 2 || output->shape[0] != spectrogram_bins || output->shape[1] != bins) return;

    double mel_lower = hz_to_mel(lower);
    double mel_upper = hz_to_mel(upper);

    for (int i = 0; i < bins; i++) {
        double left_mel = mel_lower + (mel_upper - mel_lower) * (double)i / (double)(bins + 1);
        double center_mel = mel_lower + (mel_upper - mel_lower) * (double)(i + 1) / (double)(bins + 1);
        double right_mel = mel_lower + (mel_upper - mel_lower) * (double)(i + 2) / (double)(bins + 1);

        int left = (int)floor((double)(dft_len + 1) * mel_to_hz(left_mel) / (double)rate);
        int center = (int)floor((double)(dft_len + 1) * mel_to_hz(center_mel) / (double)rate);
        int right = (int)floor((double)(dft_len + 1) * mel_to_hz(right_mel) / (double)rate);

        if (left < 0) left = 0;
        if (center < 0) center = 0;
        if (center > spectrogram_bins - 1) center = spectrogram_bins - 1;
        if (right < 0) right = 0;
        if (right > spectrogram_bins) right = spectrogram_bins;

        if (center == left && center >= 0 && center < spectrogram_bins) {
            set_tensor_value_from_float(output, (size_t)center * bins + i, 1.0);
        } else {
            for (int j = left; j <= center && j < spectrogram_bins; j++) {
                if (j >= 0) {
                    double value = (double)(j - left) / (double)(center - left);
                    set_tensor_value_from_float(output, (size_t)j * bins + i, value);
                }
            }
        }

        if (right > center) {
            for (int j = center; j < right && j < spectrogram_bins; j++) {
                if (j >= 0) {
                    double value = (double)(right - j) / (double)(right - center);
                    set_tensor_value_from_float(output, (size_t)j * bins + i, value);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `complex_tensor_index` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static size_t complex_tensor_index(const Tensor* tensor, const int* coords, int component) {
    int complex_rank = tensor->ndim - 1;
    size_t idx = 0;
    for (int d = 0; d < complex_rank; d++) {
        idx = idx * (size_t)tensor->shape[d] + (size_t)coords[d];
    }
    return idx * (size_t)tensor->shape[complex_rank] + (size_t)component;
}

// Egor Izmaylov: Function `get_complex_value` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static void get_complex_value(const Tensor* tensor, const int* coords, double* real, double* imag) {
    *real = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 0));
    *imag = 0.0;
    if (tensor->shape[tensor->ndim - 1] == 2) {
        *imag = get_value_as_double(tensor, complex_tensor_index(tensor, coords, 1));
    }
}

// Egor Izmaylov: Function `normalize_complex_axis` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static int normalize_complex_axis(int axis, int complex_rank) {
    if (axis < 0) axis += complex_rank + 1;
    return axis;
}

// Egor Izmaylov: Function `dft_forward` is the C backend entry point for the dft operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void dft_forward(const Tensor* input, Tensor* output, int axis, int inverse, int onesided, int dft_length) {
    if (!input || !output || !input->data || !output->data) return;
    if (input->ndim < 2 || output->ndim != input->ndim || input->ndim > MAX_NDIM) return;
    int complex_rank = input->ndim - 1;
    int input_complex_dim = input->shape[complex_rank];
    int output_complex_dim = output->shape[complex_rank];
    if (input_complex_dim != 1 && input_complex_dim != 2) return;
    if (output_complex_dim != 1 && output_complex_dim != 2) return;
    axis = normalize_complex_axis(axis, complex_rank);
    if (axis < 0 || axis >= complex_rank || dft_length <= 0) return;

    for (int d = 0; d < complex_rank; d++) {
        if (d != axis && input->shape[d] != output->shape[d]) return;
    }

    int input_axis_len = input->shape[axis];
    int output_axis_len = output->shape[axis];
    size_t vector_total = 1;
    for (int d = 0; d < complex_rank; d++) {
        if (d != axis) vector_total *= (size_t)output->shape[d];
    }

    _Pragma("omp parallel for collapse(2)")
    for (size_t vector_id = 0; vector_id < vector_total; vector_id++) {
        for (int k = 0; k < output_axis_len; k++) {
            int in_coords[MAX_NDIM] = {0};
            int out_coords[MAX_NDIM] = {0};
            size_t rem = vector_id;
            for (int d = complex_rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                int dim = output->shape[d];
                int coord = (int)(rem % (size_t)dim);
                rem /= (size_t)dim;
                in_coords[d] = coord;
                out_coords[d] = coord;
            }
            out_coords[axis] = k;

            if (inverse && onesided) {
                double real_sum = 0.0;
                int max_freq = dft_length / 2;
                for (int f = 0; f < input_axis_len; f++) {
                    if (f > max_freq) continue;
                    in_coords[axis] = f;
                    double xr, xi;
                    get_complex_value(input, in_coords, &xr, &xi);
                    double angle = TWO_PI * (double)f * (double)k / (double)dft_length;
                    double contribution = xr * cos(angle) - xi * sin(angle);
                    if (f != 0 && !(dft_length % 2 == 0 && f == dft_length / 2)) {
                        contribution *= 2.0;
                    }
                    real_sum += contribution;
                }
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum / (double)dft_length);
                continue;
            }

            double real_sum = 0.0;
            double imag_sum = 0.0;
            double sign = inverse ? 1.0 : -1.0;
            for (int n = 0; n < dft_length; n++) {
                double xr = 0.0;
                double xi = 0.0;
                if (n < input_axis_len) {
                    in_coords[axis] = n;
                    get_complex_value(input, in_coords, &xr, &xi);
                }
                double angle = sign * TWO_PI * (double)k * (double)n / (double)dft_length;
                double ca = cos(angle);
                double sa = sin(angle);
                real_sum += xr * ca - xi * sa;
                imag_sum += xr * sa + xi * ca;
            }
            if (inverse) {
                real_sum /= (double)dft_length;
                imag_sum /= (double)dft_length;
            }
            set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum);
            if (output_complex_dim == 2) {
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 1), imag_sum);
            }
        }
    }
}

// Egor Izmaylov: Function `stft_forward` is the C backend entry point for the stft operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void stft_forward(const Tensor* signal, const Tensor* window, Tensor* output,
                  int frame_step, int frame_length, int onesided) {
    if (!signal || !output || !signal->data || !output->data) return;
    if (signal->ndim < 2 || output->ndim != signal->ndim + 1 || signal->ndim + 1 > MAX_NDIM) return;
    if (frame_step <= 0 || frame_length <= 0) return;
    int signal_complex_rank = signal->ndim - 1;
    int output_complex_rank = output->ndim - 1;
    int prefix_rank = signal->ndim - 2;
    int signal_len = signal->shape[signal_complex_rank - 1];
    int signal_complex_dim = signal->shape[signal_complex_rank];
    int output_complex_dim = output->shape[output_complex_rank];
    if (signal_complex_dim != 1 && signal_complex_dim != 2) return;
    if (output_complex_dim != 2) return;
    for (int d = 0; d < prefix_rank; d++) {
        if (signal->shape[d] != output->shape[d]) return;
    }
    int n_frames = output->shape[prefix_rank];
    int bins = output->shape[prefix_rank + 1];
    int expected_bins = onesided ? frame_length / 2 + 1 : frame_length;
    if (bins != expected_bins) return;
    if (window && window->data && window->size < (size_t)frame_length) return;

    size_t prefix_total = 1;
    for (int d = 0; d < prefix_rank; d++) prefix_total *= (size_t)signal->shape[d];

    _Pragma("omp parallel for collapse(3)")
    for (size_t prefix_id = 0; prefix_id < prefix_total; prefix_id++) {
        for (int frame = 0; frame < n_frames; frame++) {
            for (int k = 0; k < bins; k++) {
                int sig_coords[MAX_NDIM] = {0};
                int out_coords[MAX_NDIM] = {0};
                size_t rem = prefix_id;
                for (int d = prefix_rank - 1; d >= 0; d--) {
                    int dim = signal->shape[d];
                    int coord = (int)(rem % (size_t)dim);
                    rem /= (size_t)dim;
                    sig_coords[d] = coord;
                    out_coords[d] = coord;
                }
                out_coords[prefix_rank] = frame;
                out_coords[prefix_rank + 1] = k;

                double real_sum = 0.0;
                double imag_sum = 0.0;
                for (int n = 0; n < frame_length; n++) {
                    int signal_pos = frame * frame_step + n;
                    double xr = 0.0;
                    double xi = 0.0;
                    if (signal_pos >= 0 && signal_pos < signal_len) {
                        sig_coords[prefix_rank] = signal_pos;
                        get_complex_value(signal, sig_coords, &xr, &xi);
                    }
                    double win = 1.0;
                    if (window && window->data) {
                        win = get_value_as_double(window, (size_t)n);
                    }
                    xr *= win;
                    xi *= win;
                    double angle = -TWO_PI * (double)k * (double)n / (double)frame_length;
                    double ca = cos(angle);
                    double sa = sin(angle);
                    real_sum += xr * ca - xi * sa;
                    imag_sum += xr * sa + xi * ca;
                }
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum);
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 1), imag_sum);
            }
        }
    }
}

// Egor Izmaylov: Function `recurrent_alpha` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double recurrent_alpha(const float* values, int index, double default_value) {
    if (!values) return default_value;
    float value = values[index];
    return isnan(value) ? default_value : (double)value;
}

// Egor Izmaylov: Function `recurrent_clip` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double recurrent_clip(double value, float clip, int has_clip) {
    if (!has_clip) return value;
    if (value > (double)clip) return (double)clip;
    if (value < -(double)clip) return -(double)clip;
    return value;
}

// Egor Izmaylov: Function `recurrent_activation` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `recurrent_activation_code` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static int recurrent_activation_code(const int* activations, int num_activations, int index, int default_code) {
    if (!activations || index >= num_activations) return default_code;
    return activations[index];
}

// Egor Izmaylov: Function `recurrent_num_dirs` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static int recurrent_num_dirs(int direction) {
    return direction == 2 ? 2 : 1;
}

// Egor Izmaylov: Function `recurrent_is_reverse` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static int recurrent_is_reverse(int direction, int dir_index) {
    return direction == 1 || (direction == 2 && dir_index == 1);
}

// Egor Izmaylov: Function `recurrent_x_index` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `recurrent_y_index` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `recurrent_sequence_active` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static int recurrent_sequence_active(const Tensor* sequence_lens, int t, int b) {
    if (!sequence_lens || !sequence_lens->data) return 1;
    return get_value_as_int64(sequence_lens, (size_t)b) > t;
}

// Egor Izmaylov: Function `rnn_forward` is the C backend entry point for the rnn operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void rnn_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 const int* activations, const float* activation_alpha,
                 const float* activation_beta, int num_activations,
                 float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    if (X->ndim != 3 || W->ndim != 3 || R->ndim != 3 || Y->ndim != 4) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[0] != num_dirs || R->shape[0] != num_dirs || W->shape[1] != hidden || R->shape[1] != hidden || R->shape[2] != hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* h_new = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !h_new) {
        free(h_state);
        free(h_new);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++) {
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    h_state[((size_t)d * batch + b) * hidden + h] =
                        get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
                }
            }
        }
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int act_code = recurrent_activation_code(activations, num_activations, d, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double pre = 0.0;
                    for (int i = 0; i < input_size; i++) {
                        pre += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                             * get_value_as_double(W, ((size_t)d * hidden + h) * input_size + i);
                    }
                    for (int hh = 0; hh < hidden; hh++) {
                        pre += h_state[((size_t)d * batch + b) * hidden + hh]
                             * get_value_as_double(R, ((size_t)d * hidden + h) * hidden + hh);
                    }
                    if (B && B->data) {
                        pre += get_value_as_double(B, (size_t)d * 2 * hidden + h);
                        pre += get_value_as_double(B, (size_t)d * 2 * hidden + hidden + h);
                    }
                    pre = recurrent_clip(pre, clip, has_clip);
                    h_new[(size_t)b * hidden + h] = recurrent_activation(pre, act_code, activation_alpha, activation_beta, d);
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    if (active) h_state[state_idx] = h_new[(size_t)b * hidden + h];
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    if (Y_h && Y_h->data) {
        for (int d = 0; d < num_dirs; d++) {
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    size_t idx = ((size_t)d * batch + b) * hidden + h;
                    set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                }
            }
        }
    }
    free(h_state);
    free(h_new);
}

// Egor Izmaylov: Function `gru_forward` is the C backend entry point for the gru operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void gru_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 int linear_before_reset, const int* activations,
                 const float* activation_alpha, const float* activation_beta,
                 int num_activations, float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[1] != 3 * hidden || R->shape[1] != 3 * hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* z = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* reset = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* cand = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !z || !reset || !cand) {
        free(h_state); free(z); free(reset); free(cand);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    h_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int f_code = recurrent_activation_code(activations, num_activations, d * 2, 1);
        int g_code = recurrent_activation_code(activations, num_activations, d * 2 + 1, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double gate_pre[2] = {0.0, 0.0};
                    for (int gate = 0; gate < 2; gate++) {
                        for (int i = 0; i < input_size; i++) {
                            gate_pre[gate] += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                                * get_value_as_double(W, ((size_t)d * 3 * hidden + gate * hidden + h) * input_size + i);
                        }
                        for (int hh = 0; hh < hidden; hh++) {
                            gate_pre[gate] += h_state[((size_t)d * batch + b) * hidden + hh]
                                * get_value_as_double(R, ((size_t)d * 3 * hidden + gate * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            gate_pre[gate] += get_value_as_double(B, (size_t)d * 6 * hidden + gate * hidden + h);
                            gate_pre[gate] += get_value_as_double(B, (size_t)d * 6 * hidden + 3 * hidden + gate * hidden + h);
                        }
                        gate_pre[gate] = recurrent_clip(gate_pre[gate], clip, has_clip);
                    }
                    z[(size_t)b * hidden + h] = recurrent_activation(gate_pre[0], f_code, activation_alpha, activation_beta, d * 2);
                    reset[(size_t)b * hidden + h] = recurrent_activation(gate_pre[1], f_code, activation_alpha, activation_beta, d * 2);
                }
            }
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double pre = 0.0;
                    for (int i = 0; i < input_size; i++) {
                        pre += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                             * get_value_as_double(W, ((size_t)d * 3 * hidden + 2 * hidden + h) * input_size + i);
                    }
                    if (linear_before_reset) {
                        double rec = 0.0;
                        for (int hh = 0; hh < hidden; hh++) {
                            rec += h_state[((size_t)d * batch + b) * hidden + hh]
                                 * get_value_as_double(R, ((size_t)d * 3 * hidden + 2 * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) rec += get_value_as_double(B, (size_t)d * 6 * hidden + 5 * hidden + h);
                        pre += reset[(size_t)b * hidden + h] * rec;
                        if (B && B->data) pre += get_value_as_double(B, (size_t)d * 6 * hidden + 2 * hidden + h);
                    } else {
                        for (int hh = 0; hh < hidden; hh++) {
                            pre += reset[(size_t)b * hidden + hh] * h_state[((size_t)d * batch + b) * hidden + hh]
                                 * get_value_as_double(R, ((size_t)d * 3 * hidden + 2 * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            pre += get_value_as_double(B, (size_t)d * 6 * hidden + 2 * hidden + h);
                            pre += get_value_as_double(B, (size_t)d * 6 * hidden + 5 * hidden + h);
                        }
                    }
                    pre = recurrent_clip(pre, clip, has_clip);
                    cand[(size_t)b * hidden + h] = recurrent_activation(pre, g_code, activation_alpha, activation_beta, d * 2 + 1);
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    double h_old = h_state[state_idx];
                    double h_new = (1.0 - z[(size_t)b * hidden + h]) * cand[(size_t)b * hidden + h] + z[(size_t)b * hidden + h] * h_old;
                    if (active) h_state[state_idx] = h_new;
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    if (Y_h && Y_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++) {
                    size_t idx = ((size_t)d * batch + b) * hidden + h;
                    set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                }
    }
    free(h_state); free(z); free(reset); free(cand);
}

// Egor Izmaylov: Function `lstm_forward` is the C backend entry point for the lstm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void lstm_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                  const Tensor* sequence_lens, const Tensor* initial_h,
                  const Tensor* initial_c, const Tensor* P,
                  Tensor* Y, Tensor* Y_h, Tensor* Y_c, int hidden_size,
                  int direction, int layout, int input_forget,
                  const int* activations, const float* activation_alpha,
                  const float* activation_beta, int num_activations,
                  float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[1] != 4 * hidden || R->shape[1] != 4 * hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* c_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* h_next = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* c_next = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !c_state || !h_next || !c_next) {
        free(h_state); free(c_state); free(h_next); free(c_next);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    h_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
    }
    if (initial_c && initial_c->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    c_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_c, ((size_t)d * batch + b) * hidden + h);
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int f_code = recurrent_activation_code(activations, num_activations, d * 3, 1);
        int g_code = recurrent_activation_code(activations, num_activations, d * 3 + 1, 0);
        int h_code = recurrent_activation_code(activations, num_activations, d * 3 + 2, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double gates[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int gate = 0; gate < 4; gate++) {
                        for (int i = 0; i < input_size; i++) {
                            gates[gate] += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                                * get_value_as_double(W, ((size_t)d * 4 * hidden + gate * hidden + h) * input_size + i);
                        }
                        for (int hh = 0; hh < hidden; hh++) {
                            gates[gate] += h_state[((size_t)d * batch + b) * hidden + hh]
                                * get_value_as_double(R, ((size_t)d * 4 * hidden + gate * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            gates[gate] += get_value_as_double(B, (size_t)d * 8 * hidden + gate * hidden + h);
                            gates[gate] += get_value_as_double(B, (size_t)d * 8 * hidden + 4 * hidden + gate * hidden + h);
                        }
                    }
                    double c_prev = c_state[((size_t)d * batch + b) * hidden + h];
                    double p_i = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + h) : 0.0;
                    double p_o = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + hidden + h) : 0.0;
                    double p_f = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + 2 * hidden + h) : 0.0;
                    double i_gate = recurrent_activation(recurrent_clip(gates[0] + p_i * c_prev, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    double f_gate = input_forget ? (1.0 - i_gate) : recurrent_activation(recurrent_clip(gates[2] + p_f * c_prev, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    double c_bar = recurrent_activation(recurrent_clip(gates[3], clip, has_clip), g_code, activation_alpha, activation_beta, d * 3 + 1);
                    double c_val = f_gate * c_prev + i_gate * c_bar;
                    double o_gate = recurrent_activation(recurrent_clip(gates[1] + p_o * c_val, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    h_next[(size_t)b * hidden + h] = o_gate * recurrent_activation(c_val, h_code, activation_alpha, activation_beta, d * 3 + 2);
                    c_next[(size_t)b * hidden + h] = c_val;
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    if (active) {
                        h_state[state_idx] = h_next[(size_t)b * hidden + h];
                        c_state[state_idx] = c_next[(size_t)b * hidden + h];
                    }
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    for (int d = 0; d < num_dirs; d++)
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < hidden; h++) {
                size_t idx = ((size_t)d * batch + b) * hidden + h;
                if (Y_h && Y_h->data) set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                if (Y_c && Y_c->data) set_tensor_value_from_float(Y_c, idx, c_state[idx]);
            }
    free(h_state); free(c_state); free(h_next); free(c_next);
}

// Egor Izmaylov: Function `multinomial_forward` is the C backend entry point for the multinomial operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void multinomial_forward(const Tensor* input, Tensor* output, int sample_size, uint32_t seed) {
    if (!input || !output || input->ndim != 2 || output->ndim != 2 || sample_size < 0) return;
    int batch = input->shape[0];
    int classes = input->shape[1];
    if (output->shape[0] != batch || output->shape[1] != sample_size) return;

    for (int row = 0; row < batch; row++) {
        double total = 0.0;
        for (int c = 0; c < classes; c++) {
            double p = get_value_as_double(input, (size_t)row * classes + c);
            if (p > 0.0) total += p;
        }
        if (total <= 0.0) continue;

        uint32_t state = seed ? (seed + (uint32_t)row * 747796405u) : (uint32_t)time(NULL) + (uint32_t)row;
        for (int sample = 0; sample < sample_size; sample++) {
            uint32_t r = simple_lcg(&state);
            double threshold = ((double)r / 2147483648.0) * total;
            double cumulative = 0.0;
            int selected = classes - 1;
            for (int c = 0; c < classes; c++) {
                double p = get_value_as_double(input, (size_t)row * classes + c);
                if (p <= 0.0) continue;
                cumulative += p;
                if (threshold < cumulative) {
                    selected = c;
                    break;
                }
            }
            set_tensor_value_from_int(output, (size_t)row * sample_size + sample, selected);
        }
    }
}

// Egor Izmaylov: Function `loss_spatial_size` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static size_t loss_spatial_size(const Tensor* input) {
    size_t spatial = 1;
    for (int i = 2; i < input->ndim; i++) spatial *= (size_t)input->shape[i];
    return spatial;
}

// Egor Izmaylov: Function `loss_target_weight` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double loss_target_weight(const Tensor* weight, int64_t cls) {
    if (!weight) return 1.0;
    return get_value_as_double(weight, (size_t)cls);
}

// Egor Izmaylov: Function `negative_log_likelihood_loss_forward` is the C backend entry point for the negative log likelihood loss operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void negative_log_likelihood_loss_forward(const Tensor* input, const Tensor* target, const Tensor* weight,
                                          Tensor* output, int reduction, int has_ignore_index, int64_t ignore_index) {
    if (!input || !target || !output || input->ndim < 2) return;
    int batch = input->shape[0];
    int classes = input->shape[1];
    size_t spatial = loss_spatial_size(input);
    size_t total = (size_t)batch * spatial;
    double sum = 0.0;
    double denom = 0.0;

    for (size_t i = 0; i < total; i++) {
        int64_t cls = get_value_as_int64(target, i);
        double weighted_loss = 0.0;
        double cur_weight = 0.0;
        if (!(has_ignore_index && cls == ignore_index) && cls >= 0 && cls < classes) {
            cur_weight = loss_target_weight(weight, cls);
            size_t n = i / spatial;
            size_t s = i % spatial;
            size_t input_idx = n * (size_t)classes * spatial + (size_t)cls * spatial + s;
            weighted_loss = -get_value_as_double(input, input_idx) * cur_weight;
        }

        if (reduction == 0) {
            set_tensor_value_from_float(output, i, weighted_loss);
        } else {
            sum += weighted_loss;
            if (weight || has_ignore_index) denom += cur_weight;
            else denom += 1.0;
        }
    }

    if (reduction == 2) {
        set_tensor_value_from_float(output, 0, sum);
    } else if (reduction == 1) {
        set_tensor_value_from_float(output, 0, denom == 0.0 ? NAN : sum / denom);
    }
}

// Egor Izmaylov: Function `softmax_cross_entropy_loss_forward` is the C backend entry point for the softmax cross entropy loss operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void softmax_cross_entropy_loss_forward(const Tensor* scores, const Tensor* labels, const Tensor* weights,
                                        Tensor* loss_output, Tensor* log_prob_output,
                                        int reduction, int has_ignore_index, int64_t ignore_index) {
    if (!scores || !labels || !loss_output || scores->ndim < 2) return;
    int batch = scores->shape[0];
    int classes = scores->shape[1];
    size_t spatial = loss_spatial_size(scores);
    double loss_sum = 0.0;
    double denom = 0.0;

    for (size_t n = 0; n < (size_t)batch; n++) {
        for (size_t s = 0; s < spatial; s++) {
            double max_val = -INFINITY;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                double value = get_value_as_double(scores, idx);
                if (value > max_val) max_val = value;
            }

            double exp_sum = 0.0;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                exp_sum += exp(get_value_as_double(scores, idx) - max_val);
            }
            double log_sum = log(exp_sum);

            size_t flat_target = n * spatial + s;
            int64_t cls = get_value_as_int64(labels, flat_target);
            double selected_loss = 0.0;
            double cur_weight = 0.0;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                double log_prob = get_value_as_double(scores, idx) - max_val - log_sum;
                if (log_prob_output) set_tensor_value_from_float(log_prob_output, idx, log_prob);
                if (c == cls && !(has_ignore_index && cls == ignore_index)) {
                    cur_weight = loss_target_weight(weights, cls);
                    selected_loss = -log_prob * cur_weight;
                }
            }

            if (reduction == 0) {
                set_tensor_value_from_float(loss_output, flat_target, selected_loss);
            } else {
                loss_sum += selected_loss;
                if (!(has_ignore_index && cls == ignore_index)) {
                    if (weights) denom += cur_weight;
                    else denom += 1.0;
                }
            }
        }
    }

    if (reduction == 2) {
        set_tensor_value_from_float(loss_output, 0, loss_sum);
    } else if (reduction == 1) {
        set_tensor_value_from_float(loss_output, 0, denom == 0.0 ? NAN : loss_sum / denom);
    }
}

// Egor Izmaylov: Function `nms_box_corners` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `nms_iou` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `non_max_suppression_forward` is the C backend entry point for the non max suppression operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
int non_max_suppression_forward(const Tensor* boxes, const Tensor* scores, Tensor* output,
                                int max_output_boxes_per_class, float iou_threshold,
                                float score_threshold, int center_point_box) {
    if (!boxes || !scores || !output) return 0;
    if (boxes->ndim != 3 || scores->ndim != 3 || boxes->shape[2] != 4) return 0;
    int batch_count = boxes->shape[0];
    int num_boxes = boxes->shape[1];
    int class_count = scores->shape[1];
    if (scores->shape[0] != batch_count || scores->shape[2] != num_boxes || max_output_boxes_per_class <= 0) return 0;

    int* candidates = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    int* kept = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    if (!candidates || !kept) {
        free(candidates);
        free(kept);
        return 0;
    }

    int out_rows = 0;
    for (int b = 0; b < batch_count; b++) {
        for (int cls = 0; cls < class_count; cls++) {
            int candidate_count = 0;
            for (int box = 0; box < num_boxes; box++) {
                size_t score_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)box;
                double score = get_value_as_double(scores, score_idx);
                if (score >= (double)score_threshold) {
                    candidates[candidate_count++] = box;
                }
            }

            for (int i = 1; i < candidate_count; i++) {
                int current = candidates[i];
                size_t current_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)current;
                double current_score = get_value_as_double(scores, current_idx);
                int j = i - 1;
                while (j >= 0) {
                    int prev = candidates[j];
                    size_t prev_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)prev;
                    double prev_score = get_value_as_double(scores, prev_idx);
                    if (prev_score >= current_score) break;
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = current;
            }

            int kept_count = 0;
            for (int i = 0; i < candidate_count && kept_count < max_output_boxes_per_class; i++) {
                int candidate = candidates[i];
                int suppress = 0;
                for (int k = 0; k < kept_count; k++) {
                    if (nms_iou(boxes, b, candidate, kept[k], center_point_box) > (double)iou_threshold) {
                        suppress = 1;
                        break;
                    }
                }
                if (!suppress) {
                    kept[kept_count++] = candidate;
                    if ((size_t)(out_rows + 1) * 3 <= output->size) {
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 0, b);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 1, cls);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 2, candidate);
                    }
                    out_rows++;
                }
            }
        }
    }

    free(candidates);
    free(kept);
    return out_rows;
}

// Egor Izmaylov: Function `grid_denormalize` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double grid_denormalize(double coord, int length, int align_corners) {
    if (align_corners) {
        return (coord + 1.0) * (double)(length - 1) / 2.0;
    }
    return ((coord + 1.0) * (double)length - 1.0) / 2.0;
}

// Egor Izmaylov: Function `grid_reflect_coordinate` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static double grid_reflect_coordinate(double coord, double low, double high) {
    if (high <= low) return low;
    double span = high - low;
    double value = fabs(fmod(coord - low, 2.0 * span));
    if (value > span) value = 2.0 * span - value;
    return value + low;
}

// Egor Izmaylov: Function `grid_sample_coordinate` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `grid_get_pixel_2d` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `grid_bilinear_sample_2d` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `grid_cubic_coefficients` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static void grid_cubic_coefficients(double t, double coeffs[4]) {
    double alpha = -0.75;
    double x = fabs(t);
    coeffs[0] = ((alpha * (x + 1.0) - 5.0 * alpha) * (x + 1.0) + 8.0 * alpha) * (x + 1.0) - 4.0 * alpha;
    coeffs[1] = ((alpha + 2.0) * x - (alpha + 3.0)) * x * x + 1.0;
    coeffs[2] = ((alpha + 2.0) * (1.0 - x) - (alpha + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    coeffs[3] = ((alpha * (2.0 - x) - 5.0 * alpha) * (2.0 - x) + 8.0 * alpha) * (2.0 - x) - 4.0 * alpha;
}

// Egor Izmaylov: Function `grid_bicubic_sample_2d` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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

// Egor Izmaylov: Function `grid_sample_forward` is the C backend entry point for the grid sample operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void grid_sample_forward(const Tensor* input, const Tensor* grid, Tensor* output,
                         int mode, int padding_mode, int align_corners) {
    if (!input || !grid || !output) return;
    if (input->ndim != 4 || grid->ndim != 4 || output->ndim != 4 || grid->shape[3] != 2) return;
    int n_batches = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];
    int out_h = grid->shape[1];
    int out_w = grid->shape[2];
    if (grid->shape[0] != n_batches || output->shape[0] != n_batches || output->shape[1] != channels ||
        output->shape[2] != out_h || output->shape[3] != out_w) return;

    _Pragma("omp parallel for collapse(4)")
    for (int n = 0; n < n_batches; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    size_t grid_idx = ((size_t)n * out_h * out_w * 2) + ((size_t)oy * out_w * 2) + ((size_t)ox * 2);
                    double x_norm = get_value_as_double(grid, grid_idx);
                    double y_norm = get_value_as_double(grid, grid_idx + 1);
                    double in_x = grid_denormalize(x_norm, width, align_corners);
                    double in_y = grid_denormalize(y_norm, height, align_corners);
                    double value;
                    if (mode == 1) {
                        double sy = nearbyint(grid_sample_coordinate(in_y, height, padding_mode, align_corners));
                        double sx = nearbyint(grid_sample_coordinate(in_x, width, padding_mode, align_corners));
                        value = grid_get_pixel_2d(input, n, c, sy, sx, padding_mode, align_corners);
                    } else if (mode == 2) {
                        value = grid_bicubic_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    } else {
                        value = grid_bilinear_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    }
                    size_t out_idx = ((size_t)n * channels * out_h * out_w)
                                   + ((size_t)c * out_h * out_w)
                                   + ((size_t)oy * out_w)
                                   + (size_t)ox;
                    set_tensor_value_from_float(output, out_idx, value);
                }
            }
        }
    }
}

// Egor Izmaylov: Function `lrn_forward` is the C backend entry point for the lrn operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void lrn_forward(const Tensor* input, Tensor* output, int size, float alpha, float beta, float bias) {
    if (!input || !output || input->ndim < 3 || input->size != output->size || size <= 0) return;

    int channels = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    size_t batch_size = input->shape[0];
    int lower = (size - 1) / 2;
    int upper = size - 1 - lower;

    _Pragma("omp parallel for collapse(2)")
    for (size_t n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            int begin = c - lower;
            int end = c + upper + 1;
            if (begin < 0) begin = 0;
            if (end > channels) end = channels;

            for (size_t s = 0; s < spatial_size; s++) {
                double square_sum = 0.0;
                for (int cc = begin; cc < end; cc++) {
                    size_t idx = (n * (size_t)channels + (size_t)cc) * spatial_size + s;
                    double val = get_value_as_double(input, idx);
                    square_sum += val * val;
                }
                size_t out_idx = (n * (size_t)channels + (size_t)c) * spatial_size + s;
                double x = get_value_as_double(input, out_idx);
                double denom = pow((double)bias + ((double)alpha / (double)size) * square_sum, (double)beta);
                set_tensor_value_from_float(output, out_idx, x / denom);
            }
        }
    }
}

// Egor Izmaylov: Function `mean_variance_normalization_forward` is the C backend entry point for the mean variance normalization operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void mean_variance_normalization_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (!input || !output || !params || input->size != output->size) return;

    int ndim = input->ndim;
    int* axes = params->axes;
    int num_axes = params->num_axes;
    if (ndim > MAX_NDIM || num_axes < 1) return;

    size_t reduce_total_steps = 1;
    for (int i = 0; i < num_axes; i++) {
        if (axes[i] < 0 || axes[i] >= ndim) return;
        reduce_total_steps *= input->shape[axes[i]];
    }
    if (reduce_total_steps == 0) return;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        int base_coords[MAX_NDIM] = {0};
        get_coords_from_index(i, base_coords, input->shape, ndim);

        double sum = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            int coords[MAX_NDIM];
            memcpy(coords, base_coords, ndim * sizeof(int));
            size_t temp_r = r;
            for (int k = num_axes - 1; k >= 0; k--) {
                int axis_idx = axes[k];
                int dim_size = input->shape[axis_idx];
                coords[axis_idx] = temp_r % dim_size;
                temp_r /= dim_size;
            }
            size_t idx = get_index_from_coords(coords, input->shape, ndim);
            sum += get_value_as_double(input, idx);
        }

        double mean = sum / (double)reduce_total_steps;
        double sq_sum = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            int coords[MAX_NDIM];
            memcpy(coords, base_coords, ndim * sizeof(int));
            size_t temp_r = r;
            for (int k = num_axes - 1; k >= 0; k--) {
                int axis_idx = axes[k];
                int dim_size = input->shape[axis_idx];
                coords[axis_idx] = temp_r % dim_size;
                temp_r /= dim_size;
            }
            size_t idx = get_index_from_coords(coords, input->shape, ndim);
            double diff = get_value_as_double(input, idx) - mean;
            sq_sum += diff * diff;
        }

        double variance = sq_sum / (double)reduce_total_steps;
        double x = get_value_as_double(input, i);
        set_tensor_value_from_float(output, i, (x - mean) / sqrt(variance));
    }
}

// Egor Izmaylov: Function `eye_like_forward` is the C backend entry point for the eye like operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void eye_like_forward(Tensor* output, int k) {
    if (!output || output->ndim != 2) return;
    int cols = output->shape[1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int row = (int)(i / (size_t)cols);
        int col = (int)(i % (size_t)cols);
        double value = (col == row + k) ? 1.0 : 0.0;
        set_tensor_value_from_float(output, i, value);
    }
}

// Clip：支持全广播
// 调用此函数前，Python 端已将 input, min_t, max_t 广播为相同形状
// Egor Izmaylov: Function `clip_forward` is the C backend entry point for the clip operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void clip_forward(const Tensor* input, Tensor* output, const Tensor* min_t, const Tensor* max_t) {
    if (!input || !output) return;
    
    // 检查指针是否存在，避免空指针解引用
    int has_min = (min_t && min_t->data);
    int has_max = (max_t && max_t->data);

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

// MatMul 实现 (无加速)
// Egor Izmaylov: Function `matmul_forward` is the C backend entry point for the matmul operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void matmul_forward(const Tensor* A, const Tensor* B, Tensor* Y) {
    if (!A || !B || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }
    if (ndim < 2) return; // 至少是 2D
    int K = A->shape[A->ndim - 1];
    #pragma omp parallel for
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0}; // 最大 16 维
        get_coords_from_index(i, coords, Y->shape, ndim);
        // 当前计算的是 Y[..., m, n]
        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        double sum = 0.0;
        // 内积循环 K
        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim; // 维度对齐偏移量
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;       // 最后一维 K
                else if (d == A->ndim - 2) val = m;  // 倒数第二维 M
                else {
                    // Batch 维
                    int y_dim_idx = d + offset_a;
                    // 如果 A 在此维是 1，则广播取 0；否则跟随 Y 的坐标
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += val * stride_a;
                stride_a *= A->shape[d];
            }
            // 计算 B 的索引 (逻辑同上)
            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;       // 最后一维 N
                else if (d == B->ndim - 2) val = k;  // 倒数第二维 K
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += val * stride_b;
                stride_b *= B->shape[d];
            }
            // 混合精度计算核心：
            // get_value_as_double 自动处理了 float16/bfloat16/float8 到 double 的提升
            double val_a = get_value_as_double(A, idx_a);
            double val_b = get_value_as_double(B, idx_b);
            sum += val_a * val_b;
        }
        // 结果存回
        set_tensor_value_from_float(Y, i, sum);
    }
}

// Egor Izmaylov: Function `matmul_integer_forward` is the C backend entry point for the matmul integer operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `qlinear_matmul_forward` is the C backend entry point for the qlinear matmul operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Gather 实现
// Egor Izmaylov: Function `gather_forward` is the C backend entry point for the gather operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void gather_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;
    
    int ndim_data = data->ndim;
    int ndim_indices = indices->ndim;
    int ndim_out = output->ndim;
    
    if (axis < 0) axis += ndim_data;
    if (axis < 0 || axis >= ndim_data) return;

    int axis_dim_limit = data->shape[axis];

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM]; // 偷懒做法，最大维度不超过8
        int data_coords[MAX_NDIM];
        int indices_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, ndim_out);
        for (int j = 0; j < ndim_indices; j++) {
            indices_coords[j] = out_coords[axis + j];
        }
        
        size_t idx_idx = get_index_from_coords(indices_coords, indices->shape, ndim_indices);
        int64_t index_val = get_value_as_int64(indices, idx_idx);

        if (index_val < 0) index_val += axis_dim_limit;      
        if (index_val < 0 || index_val >= axis_dim_limit) index_val = 0; 
        
        for (int j = 0; j < axis; j++) {
            data_coords[j] = out_coords[j];
        }
        data_coords[axis] = (int)index_val;
        for (int j = axis + 1; j < ndim_data; j++) {
            data_coords[j] = out_coords[j - 1 + ndim_indices];
        }
        
        size_t data_idx = get_index_from_coords(data_coords, data->shape, ndim_data);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// Expand 实现
// Egor Izmaylov: Function `expand_forward` is the C backend entry point for the expand operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void expand_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim_in = input->ndim;
    int ndim_out = output->ndim;
    
    // 维度差 
    int offset = ndim_out - ndim_in;

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, ndim_out);
        
        // 映射回输入坐标
        for (int d = 0; d < ndim_in; d++) {
            int out_dim_idx = d + offset; // 对应输出的维度索引
            // 如果输入在该维度是1，则坐标固定为0（广播）；否则随输出变化
            if (input->shape[d] == 1) {
                in_coords[d] = 0;
            } else {
                in_coords[d] = out_coords[out_dim_idx];
            }
        }
        
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim_in);
        copy_tensor_element(output, i, input, in_idx);
    }
}

// Shape 实现
// Egor Izmaylov: Function `shape_forward` is the C backend entry point for the shape operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void shape_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    // Output 应该是一个 1D int64 张量，长度等于 input->ndim
    int64_t* out_data = (int64_t*)output->data;
    for (int i = 0; i < input->ndim; i++) {
        out_data[i] = (int64_t)input->shape[i];
    }
}

// 比较 A 和 B，结果存入 O (通常是 uint8)
// Egor Izmaylov: Macro `BINARY_COMP_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
#define BINARY_COMP_IMPL(FUNC_NAME, OPERATOR) \
void FUNC_NAME(const Tensor* A, const Tensor* B, Tensor* O) { \
    if (!A || !B || !O) return; \
    size_t loop_size = O->size; \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < loop_size; i++) { \
        double val_a = get_value_as_double(A, i); \
        double val_b = get_value_as_double(B, i); \
        /* ONNX 规范：True 为 1, False 为 0 */ \
        uint8_t res = (val_a OPERATOR val_b) ? 1 : 0; \
        ((uint8_t*)O->data)[i] = res; \
    } \
}

BINARY_COMP_IMPL(equal_forward, ==)
BINARY_COMP_IMPL(greater_forward, >)
BINARY_COMP_IMPL(less_forward, <)
BINARY_COMP_IMPL(greater_or_equal_forward, >=)
BINARY_COMP_IMPL(less_or_equal_forward, <=)

// Not: 按位取反 (bool/uint8) 或 逻辑非
// Egor Izmaylov: Function `not_forward` is the C backend entry point for the not operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void not_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        // ONNX Not 对 bool 生效，这里做逻辑非
        uint8_t res = (val == 0) ? 1 : 0; 
        ((uint8_t*)output->data)[i] = res;
    }
}

// Egor Izmaylov: Function `isnan_forward` is the C backend entry point for the isnan operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void isnan_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        uint8_t res = isnan(val) ? 1 : 0;
        ((uint8_t*)output->data)[i] = res;
    }
}

// 输入已经被看作 boolean
// Egor Izmaylov: Macro `BINARY_LOGIC_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
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
        ((uint8_t*)O->data)[i] = res; \
    } \
}

BINARY_LOGIC_IMPL(and_forward, bool_a && bool_b)
BINARY_LOGIC_IMPL(or_forward,  bool_a || bool_b)
BINARY_LOGIC_IMPL(xor_forward, bool_a != bool_b)

UNARY_OP_IMPL(sin_forward, sin(val))
UNARY_OP_IMPL(tan_forward, tan(val))
UNARY_OP_IMPL(atan_forward, atan(val))

// Egor Izmaylov: Function `sign_forward` is the C backend entry point for the sign operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Egor Izmaylov: Function `identity_forward` is the C backend entry point for the identity operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void identity_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
}

// Egor Izmaylov: Function `mod_forward` is the C backend entry point for the mod operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void mod_forward(const Tensor* A, const Tensor* B, Tensor* O, int fmod_mode) {
    if (!A || !B || !O) return;
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

// Egor Izmaylov: Function `where_forward` is the C backend entry point for the where operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void where_forward(const Tensor* Cond, const Tensor* X, const Tensor* Y, Tensor* O) {
    if (!Cond || !X || !Y || !O) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double c_val = get_value_as_double(Cond, i);
        copy_tensor_element(O, i, (c_val != 0) ? X : Y, i);
    }
}

// ConstantOfShape
// Egor Izmaylov: Function `constant_of_shape_forward` is the C backend entry point for the constant of shape operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void constant_of_shape_forward(Tensor* output, const Tensor* value) {
    if (!output) return;

    size_t loop_size = output->size;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        if (value && value->data) {
            copy_tensor_element(output, i, value, 0);
        } else {
            set_tensor_value_from_float(output, i, 0.0);
        }
    }
}

// Range
// Egor Izmaylov: Function `range_forward` is the C backend entry point for the range operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void range_forward(const Tensor* start, const Tensor* limit, const Tensor* delta, Tensor* output) {
    if (!start || !limit || !delta || !output) return;
    
    double val_start = get_value_as_double(start, 0);
    double val_delta = get_value_as_double(delta, 0);
    
    size_t loop_size = output->size;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        double res = val_start + (double)i * val_delta;
        set_tensor_value_from_float(output, i, res);
    }
}

// Tile
// 输入坐标 = 输出坐标 % 输入维度
// Egor Izmaylov: Function `tile_forward` is the C backend entry point for the tile operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void tile_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim = input->ndim;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0};
        int in_coords[MAX_NDIM] = {0};
        
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        for (int d = 0; d < ndim; d++) {
            in_coords[d] = out_coords[d] % input->shape[d];
        }

        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        copy_tensor_element(output, i, input, in_idx);
    }
}

// Pad
// mode: 0=constant, 1=reflect, 2=edge
// Egor Izmaylov: Function `pad_forward` is the C backend entry point for the pad operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void pad_forward(const Tensor* data, Tensor* output, const Tensor* pads, const Tensor* constant_value, int mode) {
    if (!data || !output || !pads) return;
    
    int ndim = data->ndim;
    
    int64_t pad_begins[MAX_NDIM];
    for (int d = 0; d < ndim; d++) {
        pad_begins[d] = get_value_as_int64(pads, d);
    }
    
    double const_val = 0.0;
    if (constant_value && constant_value->data) {
        const_val = get_value_as_double(constant_value, 0);
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0};
        int in_coords[MAX_NDIM] = {0};
        int in_bounds = 1; // 标记是否在源数据范围内
        
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        for (int d = 0; d < ndim; d++) {
            // 计算相对于源数据的坐标
            int64_t c = out_coords[d] - pad_begins[d];
            int64_t dim_len = data->shape[d];
            
            if (c >= 0 && c < dim_len) {
                // 在范围内
                in_coords[d] = (int)c;
            } else {
                // 在 Padding 区域
                if (mode == 0) { // Constant
                    in_bounds = 0;
                    break; 
                } else if (mode == 2) { // Edge
                    if (c < 0) c = 0;
                    if (c >= dim_len) c = dim_len - 1;
                    in_coords[d] = (int)c;
                } else if (mode == 1) { // Reflect
                    if (dim_len <= 1) {
                        c = 0;
                    } else {
                        int64_t M = 2 * dim_len - 2;
                        int64_t k = c % M;
                        if (k < 0) k += M;
                        if (k >= dim_len) {
                            k = M - k;
                        }
                        c = k;
                    }
                    in_coords[d] = (int)c;
                } else if (mode == 3) { // Wrap
                    if (dim_len <= 0) {
                        in_bounds = 0;
                        break;
                    }
                    c %= dim_len;
                    if (c < 0) c += dim_len;
                    in_coords[d] = (int)c;
                }
            }
        }
        
        if (in_bounds) {
            size_t in_idx = get_index_from_coords(in_coords, data->shape, ndim);
            copy_tensor_element(output, i, data, in_idx);
        } else {
            if (constant_value && constant_value->data) {
                copy_tensor_element(output, i, constant_value, 0);
            } else {
                set_tensor_value_from_float(output, i, const_val);
            }
        }
    }
}

// 检查某个轴是否在归约列表中
// Egor Izmaylov: Function `is_axis_reduced` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
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
// Egor Izmaylov: Macro `REDUCE_OP_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
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
        int out_dim_idx = 0; \
        for (int d = 0; d < ndim; d++) { \
            if (is_axis_reduced(d, axes, num_axes)) { \
                coords[d] = 0; /* 归约轴初始化为 0 */ \
            } else { \
                coords[d] = out_coords[out_dim_idx++]; \
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
REDUCE_OP_IMPL(reduce_sum_forward, 0.0, acc += val, (void)0)
// ReduceMean: Init=0, Acc+=val, Post=acc/count
REDUCE_OP_IMPL(reduce_mean_forward, 0.0, acc += val, acc /= reduce_total_steps)
// ReduceProd: Init=1, Acc*=val
REDUCE_OP_IMPL(reduce_prod_forward, 1.0, acc *= val, (void)0)
// ReduceMax: Init=-inf, Acc=max
REDUCE_OP_IMPL(reduce_max_forward, -DBL_MAX, if(val > acc) acc = val, (void)0)
// ReduceMin: Init=+inf, Acc=min
REDUCE_OP_IMPL(reduce_min_forward, DBL_MAX, if(val < acc) acc = val, (void)0)

// Egor Izmaylov: Macro `ARG_OP_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
#define ARG_OP_IMPL(FUNC_NAME, INIT_VAL, CMP_OP) \
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
        double best_val = INIT_VAL; \
        int64_t best_idx = 0; \
        \
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
        set_tensor_value_from_int(output, i, best_idx); \
    } \
}

//ArgMax和ArgMin
ARG_OP_IMPL(argmax_forward, -DBL_MAX, >)

ARG_OP_IMPL(argmin_forward, DBL_MAX, <)

// Egor Izmaylov: Macro `OMP_ATOMIC_DISPATCH` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
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
// Egor Izmaylov: Function `scatter_nd_forward` is the C backend entry point for the scatter nd operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void scatter_nd_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int reduction) {
    if (!data || !indices || !updates) return;
    
    int k = indices->shape[indices->ndim - 1]; 
    int r = data->ndim; 
    size_t loop_size = updates->size;
    int slice_ndim = r - k; 
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        int up_coords[MAX_NDIM];
        int data_coords[MAX_NDIM];
        int ind_coords[MAX_NDIM]; // indices 坐标
        
        // 反解 updates 坐标
        get_coords_from_index(i, up_coords, updates->shape, updates->ndim);
        
        // 构造 indices 的读取坐标
        for (int d = 0; d < indices->ndim - 1; d++) ind_coords[d] = up_coords[d];
        
        // 读取索引向量并构造 data 坐标前缀
        for (int j = 0; j < k; j++) {
            ind_coords[indices->ndim - 1] = j;
            size_t ind_idx = get_index_from_coords(ind_coords, indices->shape, indices->ndim);
            int64_t idx_val = get_value_as_int64(indices, ind_idx);
            
            // 处理负索引
            if (idx_val < 0) idx_val += data->shape[j];
            // 越界保护
            if (idx_val < 0) idx_val = 0;
            if (idx_val >= data->shape[j]) idx_val = data->shape[j] - 1;
            
            data_coords[j] = (int)idx_val;
        }
        
        // 补全 data 坐标后缀
        for (int j = 0; j < slice_ndim; j++) {
            data_coords[k + j] = up_coords[updates->ndim - slice_ndim + j];
        }
        
        // 计算目标索引
        size_t data_idx = get_index_from_coords(data_coords, data->shape, data->ndim);
        double val = get_value_as_double(updates, i);
        
        // 执行写入
        if (reduction == 0) {
            set_tensor_value_from_float(data, data_idx, val);
        } else if (reduction == 1) { // Add
            // 使用 switch-case 分发到具体类型以启用 omp atomic
            switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_INT32, int32_t, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_INT64, int64_t, +=)
                default: 
                    // 对于不支持 atomic 的类型，使用 critical
                    #pragma omp critical
                    {
                        double old = get_value_as_double(data, data_idx);
                        set_tensor_value_from_float(data, data_idx, old + val);
                    }
                    break;
            }
        } else if (reduction == 2) { // Mul
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, *=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, *=)
                default:
                    #pragma omp critical
                    {
                        double old = get_value_as_double(data, data_idx);
                        set_tensor_value_from_float(data, data_idx, old * val);
                    }
            }
        }
    }
}

// GatherND
// 遍历 output，根据 indices 构造 data 坐标读取数据
// Egor Izmaylov: Function `gather_nd_forward` is the C backend entry point for the gather nd operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void gather_nd_forward(const Tensor* data, const Tensor* indices, Tensor* output, int batch_dims) {
    if (!data || !indices || !output) return;
    
    int k = indices->shape[indices->ndim - 1]; // 索引向量长度
    int r = data->ndim;
    int q = indices->ndim - 1; // indices 的前缀维度
    int slice_ndim = r - k - batch_dims; // 结果切片的维度数

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int ind_coords[MAX_NDIM];
        int data_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, output->ndim);
        for (int b = 0; b < batch_dims; b++) {
            data_coords[b] = out_coords[b];
            ind_coords[b] = out_coords[b];
        }
        
        // indices 的坐标：前 batch_dims + (q - batch_dims) 来自 output
        for (int j = batch_dims; j < q; j++) {
            ind_coords[j] = out_coords[j];
        }
        
        // 读取 k 个索引值填充到 data_coords
        for (int j = 0; j < k; j++) {
            ind_coords[q] = j; // indices 最后一维
            size_t ind_idx = get_index_from_coords(ind_coords, indices->shape, indices->ndim);
            int64_t idx_val = get_value_as_int64(indices, ind_idx);
            
            // 维度偏移：data 的第 batch_dims + j 维
            int data_dim_idx = batch_dims + j;
            if (idx_val < 0) idx_val += data->shape[data_dim_idx];
            // 越界 clamp
            if (idx_val < 0) idx_val = 0;
            if (idx_val >= data->shape[data_dim_idx]) idx_val = data->shape[data_dim_idx] - 1;
            
            data_coords[data_dim_idx] = (int)idx_val;
        }
        
        // output 的最后 slice_ndim 维 对应 data 的最后 slice_ndim 维
        for (int j = 0; j < slice_ndim; j++) {
            data_coords[batch_dims + k + j] = out_coords[q + j];
        }
        
        size_t data_idx = get_index_from_coords(data_coords, data->shape, data->ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// GatherElements
// Egor Izmaylov: Function `gather_elements_forward` is the C backend entry point for the gather elements operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void gather_elements_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;
    
    int ndim = data->ndim;
    if (axis < 0) axis += ndim;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 获取 index 值
        // indices 和 output 形状相同
        int64_t idx_val = get_value_as_int64(indices, i);
        if (idx_val < 0) idx_val += data->shape[axis];
        if (idx_val < 0) idx_val = 0;
        if (idx_val >= data->shape[axis]) idx_val = data->shape[axis] - 1;
        
        // 修改 axis 维度的坐标
        coords[axis] = (int)idx_val;
        
        size_t data_idx = get_index_from_coords(coords, data->shape, ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// NonZero
// Egor Izmaylov: Function `nonzero_forward` is the C backend entry point for the nonzero operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void nonzero_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim = input->ndim;
    int64_t* out_ptr = (int64_t*)output->data; // NonZero 输出必定是 int64
    
    size_t current_col = 0;
    int coords[MAX_NDIM];
    
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        if (val != 0.0) {
            get_coords_from_index(i, coords, input->shape, ndim);
            // 写入 Output: Output 是 [ndim, N] 的矩阵
            // 转置存储：col 对应第 n 个非零元素，row 对应维度
            for (int d = 0; d < ndim; d++) {
                // index = d * N + current_col
                out_ptr[d * (output->shape[1]) + current_col] = (int64_t)coords[d];
            }
            current_col++;
        }
    }
}

// Resize
// Egor Izmaylov: Function `resize_forward` is the C backend entry point for the resize operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void resize_forward(const Tensor* input, Tensor* output, float* scales, int coord_mode, int mode, int nearest_mode) {
    if (!input || !output || !scales) return;
    
    int ndim = input->ndim;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        if (mode == 0) { 
            // --- Nearest Neighbor ---
            int in_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;
                
                // 坐标变换
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f; // half_pixel
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f; // pytorch_half_pixel
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f; // align_corners
                else x_in = x_out / scale; // asymmetric (default)
                
                // 最近邻取整策略
                int in_idx = 0;
                if (nearest_mode == 2) { 
                    // floor
                    in_idx = (int)floorf(x_in);
                } else if (nearest_mode == 3) { 
                    // ceil
                    in_idx = (int)ceilf(x_in);
                } else {
                    // round_prefer_floor
                    in_idx = (int)ceilf(x_in - 0.5f);
                }
                // 边界截断 (Clamp)
                if (in_idx < 0) in_idx = 0;
                if (in_idx >= input->shape[d]) in_idx = input->shape[d] - 1;
                in_coords[d] = in_idx;
            }
            size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
            double val = get_value_as_double(input, in_idx);
            set_tensor_value_from_float(output, i, val);
            
        } else {
            // --- Linear Interpolation (N-Linear) ---
            // 计算每个维度的浮点坐标 x_in
            float real_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f;
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f;
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f;
                else x_in = x_out / scale;
                
                if (x_in < 0.0f) x_in = 0.0f;
                if (x_in > (float)(input->shape[d] - 1)) x_in = (float)(input->shape[d] - 1);
                
                real_coords[d] = x_in;
            }
            // N-Linear 插值核心
            int num_neighbors = 1 << ndim; // 2^ndim
            double weighted_sum = 0.0;
            for (int n = 0; n < num_neighbors; n++) {
                double weight = 1.0;
                int neighbor_coords[MAX_NDIM];
                for (int d = 0; d < ndim; d++) {
                    float x = real_coords[d];
                    int lower = (int)floorf(x);
                    int upper = lower + 1;
                    if (upper >= input->shape[d]) upper = input->shape[d] - 1; 
                    // 检查当前邻居在维度 d 是取 Lower 还是 Upper
                    if ((n >> d) & 1) {
                        // 取 Upper
                        neighbor_coords[d] = upper;
                        weight *= (x - lower); 
                    } else {
                        // 取 Lower
                        neighbor_coords[d] = lower;
                        weight *= (1.0f - (x - lower)); 
                    }
                }
                size_t n_idx = get_index_from_coords(neighbor_coords, input->shape, ndim);
                double val = get_value_as_double(input, n_idx);
                weighted_sum += val * weight;
            }
            set_tensor_value_from_float(output, i, weighted_sum);
        }
    }
}

// 降序比较函数
// Egor Izmaylov: Function `compare_desc` is a qsort comparator used by ranking-style operators, preserving deterministic ordering for values and original indices.
int compare_desc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;

    int nan1 = isnan(e1->value);
    int nan2 = isnan(e2->value);
    
    if (nan1 && nan2) return (e1->index < e2->index) ? -1 : 1;
    if (nan1) return -1; 
    if (nan2) return 1; 

    if (e1->value > e2->value) return -1;
    if (e1->value < e2->value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}

// 升序比较函数
// Egor Izmaylov: Function `compare_asc` is a qsort comparator used by ranking-style operators, preserving deterministic ordering for values and original indices.
int compare_asc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;

    int nan1 = isnan(e1->value);
    int nan2 = isnan(e2->value);
    
    if (nan1 && nan2) return (e1->index < e2->index) ? -1 : 1;
    if (nan1) return 1; 
    if (nan2) return -1;

    if (e1->value < e2->value) return -1;
    if (e1->value > e2->value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}

// Egor Izmaylov: Function `topk_forward` is the C backend entry point for the topk operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void topk_forward(const Tensor* input, Tensor* values, Tensor* indices, int axis, int largest, int sorted, int K) {
    if (!input || !values || !indices) return;
    (void)sorted;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    int axis_dim = input->shape[axis];
    int outer_loops = 1;
    for (int i = 0; i < axis; i++) outer_loops *= input->shape[i];
    int inner_loops = 1;
    for (int i = axis + 1; i < ndim; i++) inner_loops *= input->shape[i];
    
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)outer_loops * inner_loops; i++) {
        // 计算当前处理的 row 的位置
        int inner_idx = i % inner_loops;
        int outer_idx = i / inner_loops;
        
        // 临时 buffer，存放该轴的所有元素
        TopKElement* buffer = (TopKElement*)malloc(axis_dim * sizeof(TopKElement));
        if (!buffer) continue;
        
        // 读取数据
        for (int k = 0; k < axis_dim; k++) {
            // 构造完整坐标的 flat index
            // Index = outer * (axis_dim * inner) + k * inner + inner_idx
            size_t idx = (size_t)outer_idx * axis_dim * inner_loops + (size_t)k * inner_loops + inner_idx;
            buffer[k].value = get_value_as_double(input, idx);
            buffer[k].index = k; // 记录原始下标
        }
        
        // 排序
        if (largest) {
            qsort(buffer, axis_dim, sizeof(TopKElement), compare_desc);
        } else {
            qsort(buffer, axis_dim, sizeof(TopKElement), compare_asc);
        }
        
        // 写入前 K 个
        int write_k = (K < axis_dim) ? K : axis_dim;
        for (int k = 0; k < write_k; k++) {
            // Output shape is same as Input except axis=K
            // OutIndex = outer * (K * inner) + k * inner + inner_idx
            size_t out_idx = (size_t)outer_idx * K * inner_loops + (size_t)k * inner_loops + inner_idx;
            
            set_tensor_value_from_float(values, out_idx, buffer[k].value);
            set_tensor_value_from_int(indices, out_idx, buffer[k].index);
        }
        free(buffer);
    }
}

// Egor Izmaylov: Function `cumsum_forward` is the C backend entry point for the cumsum operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void cumsum_forward(const Tensor* input, Tensor* output, int axis, int exclusive, int reverse) {
    if (!input || !output) return;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    int axis_dim = input->shape[axis];
    int outer_loops = 1;
    for (int i = 0; i < axis; i++) outer_loops *= input->shape[i];
    int inner_loops = 1;
    for (int i = axis + 1; i < ndim; i++) inner_loops *= input->shape[i];
    
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)outer_loops * inner_loops; i++) {
        int inner_idx = i % inner_loops;
        int outer_idx = i / inner_loops;
        
        double accumulator = 0.0;
        
        // 确定遍历方向
        int start = reverse ? axis_dim - 1 : 0;
        int end   = reverse ? -1 : axis_dim;
        int step  = reverse ? -1 : 1;
        
        for (int k = start; k != end; k += step) {
            size_t idx = (size_t)outer_idx * axis_dim * inner_loops + (size_t)k * inner_loops + inner_idx;
            double val = get_value_as_double(input, idx);
            
            if (exclusive) {
                set_tensor_value_from_float(output, idx, accumulator);
                accumulator += val;
            } else {
                accumulator += val;
                set_tensor_value_from_float(output, idx, accumulator);
            }
        }
    }
}

// Egor Izmaylov: Function `simple_lcg` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static uint32_t simple_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

// Egor Izmaylov: Function `random_uniform_like_forward` is the C backend entry point for the random uniform like operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void random_uniform_like_forward(Tensor* output, float low, float high, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    double range = high - low;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t local_state = base_seed + (uint32_t)(tid * 0x9E3779B9); 
        
        #pragma omp for
        for (size_t i = 0; i < output->size; i++) {
            uint32_t r = simple_lcg(&local_state);
            
            double r_norm = (double)r / 2147483648.0; 
            double val = low + r_norm * range;
            set_tensor_value_from_float(output, i, val);
        }
    }
}

// Egor Izmaylov: Function `einsum_forward` is the C backend entry point for the einsum operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void einsum_forward(const Tensor** inputs, int num_inputs, Tensor* output, 
                    int iter_dims, int* loop_limits, 
                    int* input_strides, int* output_strides) {
    
    // 总迭代次数
    size_t total_ops = 1;
    for (int i = 0; i < iter_dims; i++) total_ops *= loop_limits[i];
    size_t out_size = output->size;
    
    size_t elem_size = get_dtype_size(output->dtype);
    memset(output->data, 0, out_size * elem_size);
    
    // 并行化大循环
    #pragma omp parallel for
    for (size_t op = 0; op < total_ops; op++) {
        // 反解当前的循环计数器 (counters)
        // counters[d] 代表第 d 个“标签”当前的索引值
        // 假设 iter_dims 不会超过 26 (a-z)
        int counters[26]; 
        size_t temp_op = op;
        for (int d = iter_dims - 1; d >= 0; d--) {
            counters[d] = temp_op % loop_limits[d];
            temp_op /= loop_limits[d];
        }
        
        // 计算每个输入的 Flat Index
        // Index_k = Sum_d ( counters[d] * stride_k[d] )
        double product = 1.0;
        
        for (int k = 0; k < num_inputs; k++) {
            size_t in_idx = 0;
            int* cur_strides = &input_strides[k * iter_dims];
            
            for (int d = 0; d < iter_dims; d++) {
                in_idx += counters[d] * cur_strides[d];
            }
            
            product *= get_value_as_double(inputs[k], in_idx);
        }
        
        // 计算输出的 Flat Index
        size_t out_idx = 0;
        for (int d = 0; d < iter_dims; d++) {
            out_idx += counters[d] * output_strides[d];
        }
        
        #pragma omp critical
        {
            double old_val = get_value_as_double(output, out_idx);
            set_tensor_value_from_float(output, out_idx, old_val + product);
        }
    }
}

// Egor Izmaylov: Macro `UNARY_OP_WITH_ALPHA_IMPL` expands repeated C function implementations for related operators; it keeps generated entry points aligned with the ctypes ABI while avoiding duplicated loop code.
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
UNARY_OP_WITH_ALPHA_IMPL(elu_forward, (val > 0) ? val : a * (exp(val) - 1.0))

// LeakyRelu: x >= 0 ? x : alpha * x
UNARY_OP_WITH_ALPHA_IMPL(leaky_relu_forward, (val >= 0) ? val : a * val)

// ThresholdedRelu: x > alpha ? x : 0
UNARY_OP_WITH_ALPHA_IMPL(thresholded_relu_forward, (val > a) ? val : 0.0)

// Celu: x >= 0 ? x : alpha * (exp(x/alpha) - 1)
UNARY_OP_WITH_ALPHA_IMPL(celu_forward, (val >= 0) ? val : a * (exp(val / a) - 1.0))

// Selu: gamma * (x > 0 ? x : alpha * (exp(x) - 1))
// Egor Izmaylov: Function `selu_forward` is the C backend entry point for the selu operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
// Egor Izmaylov: Function `hard_sigmoid_forward` is the C backend entry point for the hard sigmoid operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Shrink: x < -lambd ? x + bias : (x > lambd ? x - bias : 0)
// Egor Izmaylov: Function `shrink_forward` is the C backend entry point for the shrink operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// 位运算逻辑
// Egor Izmaylov: Function `op_bitwise_and` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_bitwise_and(int64_t a, int64_t b) { return a & b; }
// Egor Izmaylov: Function `op_bitwise_or` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_bitwise_or(int64_t a, int64_t b) { return a | b; }
// Egor Izmaylov: Function `op_bitwise_xor` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_bitwise_xor(int64_t a, int64_t b) { return a ^ b; }
// Egor Izmaylov: Function `op_shift_left` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_shift_left(int64_t a, int64_t b) { return a << b; }
// Egor Izmaylov: Function `op_shift_right` implements shared tensor-operator helper logic in the C backend, factoring indexing, shape, random, reduction, or math details away from Python.
static inline int64_t op_shift_right(int64_t a, int64_t b) { return a >> b; }

// BitwiseAnd
// Egor Izmaylov: Function `bitwise_and_forward` is the C backend entry point for the bitwise and operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void bitwise_and_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_and); 
}

// BitwiseOr
// Egor Izmaylov: Function `bitwise_or_forward` is the C backend entry point for the bitwise or operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void bitwise_or_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_or);
}

// BitwiseXor
// Egor Izmaylov: Function `bitwise_xor_forward` is the C backend entry point for the bitwise xor operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void bitwise_xor_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_xor);
}

// BitwiseNot
// Egor Izmaylov: Function `bitwise_not_forward` is the C backend entry point for the bitwise not operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void bitwise_not_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        int64_t val = get_value_as_int64(input, i);
        int64_t res = ~val;
        set_tensor_value_from_int(output, i, res);
    }
}

// BitShift
// direction: 0=LEFT, 1=RIGHT
// Egor Izmaylov: Function `bit_shift_forward` is the C backend entry point for the bit shift operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// ReduceL1: Sum(|x|)
REDUCE_OP_IMPL(reduce_l1_forward, 0.0, acc += fabs(val), (void)0)

// ReduceL2: Sqrt(Sum(x^2))
REDUCE_OP_IMPL(reduce_l2_forward, 0.0, acc += val * val, acc = sqrt(acc))

// ReduceLogSum: Log(Sum(x))
REDUCE_OP_IMPL(reduce_log_sum_forward, 0.0, acc += val, acc = log(acc))

// ReduceLogSumExp: Log(Sum(exp(x)))，仅实现基础定义
REDUCE_OP_IMPL(reduce_log_sum_exp_forward, 0.0, acc += exp(val), acc = log(acc))

// ReduceSumSquare: Sum(x^2)
REDUCE_OP_IMPL(reduce_sum_square_forward, 0.0, acc += val * val, (void)0)

// AveragePool
// Egor Izmaylov: Function `average_pool_forward` is the C backend entry point for the average pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void average_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params, int count_include_pad) {
    if (!X || !Y || !params) return;
    int batch = X->shape[0];
    int channels = X->shape[1];
    int in_h = X->shape[2];
    int in_w = X->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];
    
    int k_h = params->kernel_shape[0];
    int k_w = params->kernel_shape[1];
    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double sum = 0.0;
                    int count = 0;
                    
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int h_in = oh * stride_h + kh * dilation_h - pad_top;
                            int w_in = ow * stride_w + kw * dilation_w - pad_left;
                            
                            int is_pad = (h_in < 0 || h_in >= in_h || w_in < 0 || w_in >= in_w);
                            
                            if (!is_pad) {
                                size_t x_idx = ((size_t)n * channels * in_h * in_w) + 
                                               ((size_t)c * in_h * in_w) + 
                                               ((size_t)h_in * in_w) + w_in;
                                sum += get_value_as_double(X, x_idx);
                                count++;
                            } else {
                                if (count_include_pad) count++;
                            }
                        }
                    }
                    size_t y_idx = ((size_t)n * channels * out_h * out_w) + 
                                   ((size_t)c * out_h * out_w) + 
                                   ((size_t)oh * out_w) + ow;
                    // 避免除以0
                    double avg = (count > 0) ? (sum / count) : 0.0;
                    set_tensor_value_from_float(Y, y_idx, avg);
                }
            }
        }
    }
}

// LpPool
// Egor Izmaylov: Function `lp_pool_forward` is the C backend entry point for the lp pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void lp_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params, int p) {
    if (!X || !Y || !params) return;
    int batch = X->shape[0];
    int channels = X->shape[1];
    int in_h = X->shape[2];
    int in_w = X->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];
    
    int k_h = params->kernel_shape[0];
    int k_w = params->kernel_shape[1];
    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double sum_pow = 0.0;
                    
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int h_in = oh * stride_h + kh * dilation_h - pad_top;
                            int w_in = ow * stride_w + kw * dilation_w - pad_left;
                            
                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                                size_t x_idx = ((size_t)n * channels * in_h * in_w) + 
                                               ((size_t)c * in_h * in_w) + 
                                               ((size_t)h_in * in_w) + w_in;
                                double val = get_value_as_double(X, x_idx);
                                sum_pow += pow(fabs(val), p);
                            }
                        }
                    }
                    size_t y_idx = ((size_t)n * channels * out_h * out_w) + 
                                   ((size_t)c * out_h * out_w) + 
                                   ((size_t)oh * out_w) + ow;
                    double res = pow(sum_pow, 1.0 / p);
                    set_tensor_value_from_float(Y, y_idx, res);
                }
            }
        }
    }
}

// GlobalAveragePool
// 假设输入是 NCHW (或至少后两维是空间维度)，如果不符合则不执行
// Egor Izmaylov: Function `global_average_pool_forward` is the C backend entry point for the global average pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void global_average_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            sum += get_value_as_double(input, offset + i);
        }
        set_tensor_value_from_float(output, n, sum / spatial_size);
    }
}

// GlobalMaxPool
// Egor Izmaylov: Function `global_max_pool_forward` is the C backend entry point for the global max pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void global_max_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double max_val = -DBL_MAX;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            if (val > max_val) max_val = val;
        }
        set_tensor_value_from_float(output, n, max_val);
    }
}

// GlobalLpPool
// Egor Izmaylov: Function `global_lp_pool_forward` is the C backend entry point for the global lp pool operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void global_lp_pool_forward(const Tensor* input, Tensor* output, int p) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum_pow = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            sum_pow += pow(fabs(val), p);
        }
        
        // p=1 时就是 Sum(|x|)，p=2 时是 L2 Norm，p=inf 时是 Max
        double res = pow(sum_pow, 1.0 / p);
        set_tensor_value_from_float(output, n, res);
    }
}

// Mean (Element-wise)
// Egor Izmaylov: Function `mean_forward` is the C backend entry point for the mean operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void mean_forward(const Tensor** inputs, int num_inputs, Tensor* output) {
    if (!inputs || !output || num_inputs < 1) return;
    size_t size = output->size;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < size; i++) {
        double sum = 0.0;
        for (int k = 0; k < num_inputs; k++) {
            sum += get_value_as_double(inputs[k], i);
        }
        set_tensor_value_from_float(output, i, sum / num_inputs);
    }
}

// Egor Izmaylov: Function `size_forward` is the C backend entry point for the size operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void size_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int64_t total_elems = (int64_t)input->size;
    set_tensor_value_from_int(output, 0, total_elems);
}

// IsInf
// Egor Izmaylov: Function `isinf_forward` is the C backend entry point for the isinf operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// OneHot
// indices: 输入索引
// values: [off_value, on_value] (2 element tensor)
// axis: 扩充的维度
// Egor Izmaylov: Function `one_hot_forward` is the C backend entry point for the one hot operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void one_hot_forward(const Tensor* indices, const Tensor* values, Tensor* output, int axis) {
    if (!indices || !values || !output) return;
    
    int out_ndim = output->ndim;
    if (axis < 0) axis += out_ndim;
    
    int depth = output->shape[axis];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int idx_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, out_ndim);
        
        int k = 0;
        for (int d = 0; d < out_ndim; d++) {
            if (d != axis) {
                idx_coords[k++] = out_coords[d];
            }
        }
        size_t idx_idx = get_index_from_coords(idx_coords, indices->shape, indices->ndim);
        int64_t target_idx = get_value_as_int64(indices, idx_idx);
        
        if (target_idx < 0) target_idx += depth;
        
        int current_depth_idx = out_coords[axis];
        
        copy_tensor_element(output, i, values, (current_depth_idx == target_idx) ? 1 : 0);
    }
}

// Tril / Triu
// Egor Izmaylov: Function `triangular_forward` is the C backend entry point for the triangular operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void triangular_forward(const Tensor* input, Tensor* output, int k, int upper) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return; 
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, input->shape, ndim);
        
        int row = coords[ndim - 2];
        int col = coords[ndim - 1];
        
        double val = get_value_as_double(input, i);
        double res = 0.0;
        
        if (upper) {
            if (col - row >= k) res = val;
            else res = 0.0;
        } else {
            if (col - row <= k) res = val;
            else res = 0.0;
        }
        set_tensor_value_from_float(output, i, res);
    }
}

// ================== Group 7: Normalization & Math Extensions 实现 ==================

// Round: round to nearest integer
UNARY_OP_IMPL(round_forward, rint(val))

// Erf: error function
UNARY_OP_IMPL(erf_forward, erf(val))

// BatchNormalization (Inference Mode)
// Y = (X - mean) / sqrt(var + eps) * scale + B
// 优化为: Y = X * A + K
// 其中 A = scale / sqrt(var + eps), K = B - mean * A
// Egor Izmaylov: Function `batch_norm_forward` is the C backend entry point for the batch norm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void batch_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        const Tensor* mean, const Tensor* var, Tensor* output, float epsilon) {
    if (!input || !scale || !B || !mean || !var || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    // 假设输入是 NCHW 或 NC
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    // 预计算通道参数，避免在内层循环重复计算 sqrt/div
    double* A_table = (double*)malloc(C * sizeof(double));
    double* K_table = (double*)malloc(C * sizeof(double));
    
    for (int c = 0; c < C; c++) {
        double s = get_value_as_double(scale, c);
        double b = get_value_as_double(B, c);
        double m = get_value_as_double(mean, c);
        double v = get_value_as_double(var, c);
        
        double inv_std = 1.0 / sqrt(v + epsilon);
        A_table[c] = s * inv_std;
        K_table[c] = b - m * A_table[c];
    }
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            double A_val = A_table[c];
            double K_val = K_table[c];
            size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
            
            for (size_t i = 0; i < spatial_size; i++) {
                double x = get_value_as_double(input, offset + i);
                double y = x * A_val + K_val;
                set_tensor_value_from_float(output, offset + i, y);
            }
        }
    }
    
    free(A_table);
    free(K_table);
}

// InstanceNormalization
// 对每个 (n, c) 切片计算均值和方差，然后归一化
// Egor Izmaylov: Function `instance_norm_forward` is the C backend entry point for the instance norm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void instance_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                           Tensor* output, float epsilon) {
    if (!input || !scale || !B || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
            
            double sum = 0.0;
            for (size_t i = 0; i < spatial_size; i++) {
                sum += get_value_as_double(input, offset + i);
            }
            double mean = sum / spatial_size;

            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < spatial_size; i++) {
                double val = get_value_as_double(input, offset + i);
                double diff = val - mean;
                sum_sq_diff += diff * diff;
            }
            double var = sum_sq_diff / spatial_size;
            double inv_std = 1.0 / sqrt(var + epsilon);
            
            double s = get_value_as_double(scale, c);
            double b = get_value_as_double(B, c);
            
            for (size_t i = 0; i < spatial_size; i++) {
                double x = get_value_as_double(input, offset + i);
                double y = (x - mean) * inv_std * s + b;
                set_tensor_value_from_float(output, offset + i, y);
            }
        }
    }
}

// LayerNormalization
// 沿着 axis 轴进行归一化 (通常 axis=-1)
// Egor Izmaylov: Function `layer_norm_forward` is the C backend entry point for the layer norm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void layer_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int axis, float epsilon) {
    if (!input || !output) return;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    // 偷懒，假设 axis 是最后一维 (ONNX LayerNorm 默认也是 -1)
    int norm_dim = input->shape[axis];
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= input->shape[i];
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        size_t offset = i * norm_dim;
        
        double sum = 0.0;
        for (int j = 0; j < norm_dim; j++) {
            sum += get_value_as_double(input, offset + j);
        }
        double mean = sum / norm_dim;
        
        double sum_sq_diff = 0.0;
        for (int j = 0; j < norm_dim; j++) {
            double val = get_value_as_double(input, offset + j);
            double diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        double var = sum_sq_diff / norm_dim;
        double inv_std = 1.0 / sqrt(var + epsilon);
        
        for (int j = 0; j < norm_dim; j++) {
            double x = get_value_as_double(input, offset + j);
            
            double s = 1.0;
            double b = 0.0;
            if (scale) s = get_value_as_double(scale, j);
            if (B) b = get_value_as_double(B, j);
            
            double y = (x - mean) * inv_std * s + b;
            set_tensor_value_from_float(output, offset + j, y);
        }
    }
}

// 获取窗函数大小
// Egor Izmaylov: Function `get_window_size` is a tensor ABI helper that converts, reads, writes, or copies values while preserving the DataType enum contract shared with Python ctypes.
static int64_t get_window_size(const Tensor* size_tensor) {
    if (!size_tensor) return 0;
    return get_value_as_int64(size_tensor, 0);
}

// Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
// Egor Izmaylov: Function `hann_window_forward` is the C backend entry point for the hann window operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void hann_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return; // 甚至不需要写入
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = 0.5 * (1.0 - cos(2.0 * PI * i / denom));
        set_tensor_value_from_float(output, i, val);
    }
}

// Hamming Window: 0.54 - 0.46 * cos(2*pi*n / (N-1))
// Egor Izmaylov: Function `hamming_window_forward` is the C backend entry point for the hamming window operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void hamming_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = 0.54 - 0.46 * cos(2.0 * PI * i / denom);
        set_tensor_value_from_float(output, i, val);
    }
}

// Blackman Window: 0.42 - 0.5*cos(...) + 0.08*cos(...)
// Egor Izmaylov: Function `blackman_window_forward` is the C backend entry point for the blackman window operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void blackman_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0); // center value usually
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double term1 = 0.5 * cos(2.0 * PI * i / denom);
        double term2 = 0.08 * cos(4.0 * PI * i / denom);
        double val = 0.42 - term1 + term2;
        set_tensor_value_from_float(output, i, val);
    }
}

// RandomNormal: Box-Muller 变换
// Egor Izmaylov: Function `random_normal_forward` is the C backend entry point for the random normal operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void random_normal_forward(Tensor* output, float mean, float scale, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // 确保种子分散
        uint32_t local_state = base_seed + (uint32_t)(tid * 0x9E3779B9); 
        
        #pragma omp for
        for (size_t i = 0; i < output->size; i++) {
            // 生成 u1, u2 在 (0, 1]
            uint32_t r1 = simple_lcg(&local_state);
            uint32_t r2 = simple_lcg(&local_state);
            
            // 避免 log(0)
            double u1 = ((double)r1 + 1.0) / 2147483649.0; 
            double u2 = ((double)r2 + 1.0) / 2147483649.0;
            
            double z0 = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
            double val = (double)mean + z0 * (double)scale;
            
            set_tensor_value_from_float(output, i, val);
        }
    }
}

// Bernoulli: 生成 0 或 1
// Egor Izmaylov: Function `bernoulli_forward` is the C backend entry point for the bernoulli operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void bernoulli_forward(const Tensor* input, Tensor* output, float seed) {
    if (!input || !output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t local_state = base_seed + (uint32_t)(tid * 0x9E3779B9);
        
        #pragma omp for
        for (size_t i = 0; i < output->size; i++) {
            double prob = get_value_as_double(input, i);
            uint32_t r = simple_lcg(&local_state);
            double r_norm = (double)r / 2147483648.0; // [0, 1)
            
            double res = (r_norm < prob) ? 1.0 : 0.0;
            
            // 类型需要匹配输出张量，这里使用通用 set_float
            set_tensor_value_from_float(output, i, res);
        }
    }
}

// Dropout (Inference Mode)
// Egor Izmaylov: Function `dropout_forward` is the C backend entry point for the dropout operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void dropout_forward(const Tensor* input, Tensor* output, float ratio, int training_mode) {
    if (!input || !output) return;
    
    // 如果是推理模式(training_mode=0)，或者是比例为0，直接复制
    if (training_mode == 0 || ratio == 0.0f) {
        size_t elem_size = get_dtype_size(input->dtype);
        // 如果输入输出类型一致且大小一致
        if (input->dtype == output->dtype && input->size == output->size) {
            memcpy(output->data, input->data, input->size * elem_size);
        } else {
            // 类型转换复制
            cast_forward(input, output);
        }
        return;
    }
    
    // 训练模式下的 Dropout (简单的随机置0)
    // 标准 Dropout 还需要 scale (val / (1-ratio)) 以保持期望值
    double scale_factor = 1.0 / (1.0 - (double)ratio);
    uint32_t base_seed = (uint32_t)time(NULL);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t local_state = base_seed + tid;
        
        #pragma omp for
        for (size_t i = 0; i < input->size; i++) {
            uint32_t r = simple_lcg(&local_state);
            double r_norm = (double)r / 2147483648.0;
            
            double val = get_value_as_double(input, i);
            if (r_norm < ratio) {
                set_tensor_value_from_float(output, i, 0.0);
            } else {
                set_tensor_value_from_float(output, i, val * scale_factor);
            }
        }
    }
}

// Gelu
UNARY_OP_IMPL(gelu_forward, 0.5 * val * (1.0 + erf(val * M_SQRT1_2)))

// Egor Izmaylov: Function `mish_forward` is the C backend entry point for the mish operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// Hardmax
// Egor Izmaylov: Function `hardmax_forward` is the C backend entry point for the hardmax operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void hardmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            int max_idx = 0;
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double res = (j == max_idx) ? 1.0 : 0.0;
                set_tensor_value_from_float(output, idx, res);
            }
        }
    }
}

// LogSoftmax: x - max - log(sum(exp(x - max)))
// Egor Izmaylov: Function `log_softmax_forward` is the C backend entry point for the log softmax operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void log_softmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) max_val = val;
            }
            
            double sum_exp = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum_exp += exp(val - max_val);
            }
            double log_sum = log(sum_exp);
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                double res = (val - max_val) - log_sum;
                set_tensor_value_from_float(output, idx, res);
            }
        }
    }
}

// LpNormalization
// y = x / ||x||_p
// Egor Izmaylov: Function `lp_normalization_forward` is the C backend entry point for the lp normalization operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void lp_normalization_forward(const Tensor* input, Tensor* output, int axis, int p) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double sum_pow = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum_pow += pow(fabs(val), p);
            }
            
            double norm = pow(sum_pow, 1.0 / p);
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                set_tensor_value_from_float(output, idx, norm == 0.0 ? 0.0 : val / norm);
            }
        }
    }
}

// DepthToSpace
// Egor Izmaylov: Function `depth_to_space_forward` is the C backend entry point for the depth to space operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void depth_to_space_forward(const Tensor* input, Tensor* output, int blocksize, int mode) {
    if (!input || !output) return;
    
    int N = input->shape[0];
    int C_out = output->shape[1];
    int H_out = output->shape[2];
    int W_out = output->shape[3];
    
    // 遍历输出坐标
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    // 反推输入坐标
                    // 输出坐标 (h, w) 对应 spatial block 中的 (dy, dx)
                    int in_h = h / blocksize;
                    int dy = h % blocksize;
                    int in_w = w / blocksize;
                    int dx = w % blocksize;
                    
                    int in_c = 0;
                    if (mode == 0) { // DCR: depth = [dy, dx, c]
                        // C dimension composed of (blocksize, blocksize, C_out)
                        in_c = (dy * blocksize + dx) * C_out + c;
                    } else { // CRD: depth = [c, dy, dx]
                        // C dimension composed of (C_out, blocksize, blocksize)
                        in_c = c * (blocksize * blocksize) + (dy * blocksize + dx);
                    }
                    
                    double val = get_val_4d_with_padding(input, n, in_c, in_h, in_w, 0.0);
                    
                    size_t out_idx = ((size_t)n * C_out * H_out * W_out) + 
                                     ((size_t)c * H_out * W_out) + 
                                     ((size_t)h * W_out) + w;
                    set_tensor_value_from_float(output, out_idx, val);
                }
            }
        }
    }
}

// SpaceToDepth
// Egor Izmaylov: Function `space_to_depth_forward` is the C backend entry point for the space to depth operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void space_to_depth_forward(const Tensor* input, Tensor* output, int blocksize) {
    if (!input || !output) return;
    
    int N = output->shape[0];
    int C_out = output->shape[1];
    int H_out = output->shape[2];
    int W_out = output->shape[3];
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            int C_in = input->shape[1];
            int in_c = c % C_in;
            int rem = c / C_in;
            int dy = rem / blocksize;
            int dx = rem % blocksize;
            
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int in_h = h * blocksize + dy;
                    int in_w = w * blocksize + dx;
                    
                    double val = get_val_4d_with_padding(input, n, in_c, in_h, in_w, 0.0);
                    
                    size_t out_idx = ((size_t)n * C_out * H_out * W_out) + 
                                     ((size_t)c * H_out * W_out) + 
                                     ((size_t)h * W_out) + w;
                    set_tensor_value_from_float(output, out_idx, val);
                }
            }
        }
    }
}

// ReverseSequence
// Egor Izmaylov: Function `reverse_sequence_forward` is the C backend entry point for the reverse sequence operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void reverse_sequence_forward(const Tensor* input, const Tensor* sequence_lens, Tensor* output, int time_axis, int batch_axis) {
    if (!input || !output || !sequence_lens) return;
    int ndim = input->ndim;
    if (time_axis < 0) time_axis += ndim;
    if (batch_axis < 0) batch_axis += ndim;
    
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
    
    size_t strides[MAX_NDIM];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) strides[i] = strides[i+1] * input->shape[i+1];
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        int b_idx = coords[batch_axis];
        int t_idx = coords[time_axis];
        
        int64_t seq_len = get_value_as_int64(sequence_lens, b_idx);
        
        if (t_idx < seq_len) {
            int old_t_idx = (int)seq_len - 1 - t_idx;
            coords[time_axis] = old_t_idx;
            
            size_t src_idx = get_index_from_coords(coords, input->shape, ndim);
            double val = get_value_as_double(input, src_idx);
            set_tensor_value_from_float(output, i, val);
        }
    }
}

// Compress
// Egor Izmaylov: Function `compress_forward` is the C backend entry point for the compress operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void compress_forward(const Tensor* input, const Tensor* condition, Tensor* output, int axis) {
    if (!input || !condition || !output) return;
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    int cond_len = condition->size;
    int* idx_map = (int*)malloc(cond_len * sizeof(int));
    int count = 0;
    for (int i = 0; i < cond_len; i++) {
        if (get_value_as_double(condition, i) != 0.0) {
            idx_map[count++] = i;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 映射 axis 坐标
        int out_axis_idx = coords[axis];
        if (out_axis_idx < count) {
            coords[axis] = idx_map[out_axis_idx]; // 替换为原坐标
            
            size_t src_idx = get_index_from_coords(coords, input->shape, ndim);
            double val = get_value_as_double(input, src_idx);
            set_tensor_value_from_float(output, i, val);
        }
    }
    
    free(idx_map);
}

// ScatterElements
// Egor Izmaylov: Function `scatter_elements_forward` is the C backend entry point for the scatter elements operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void scatter_elements_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int axis, int reduction) {
    if (!data || !indices || !updates) return;
    int ndim = data->ndim;
    if (axis < 0) axis += ndim;
    
    // 遍历 updates (和 indices 形状相同)
    size_t loop_size = updates->size;
    
    #pragma omp parallel for
    for (size_t i = 0; i < loop_size; i++) {
        int coords[MAX_NDIM];
        get_coords_from_index(i, coords, updates->shape, ndim);
        
        // 获取 index 值
        int64_t idx_val = get_value_as_int64(indices, i);
        if (idx_val < 0) idx_val += data->shape[axis];
        if (idx_val < 0) idx_val = 0;
        if (idx_val >= data->shape[axis]) idx_val = data->shape[axis] - 1;
        
        // 构造目标坐标: 除了 axis 维，其他与 updates 坐标一致
        coords[axis] = (int)idx_val;
        
        size_t data_idx = get_index_from_coords(coords, data->shape, ndim);
        double val = get_value_as_double(updates, i);
        
        if (reduction == 0) {
            set_tensor_value_from_float(data, data_idx, val);
        } else if (reduction == 1) { // Add
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_INT32, int32_t, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_INT64, int64_t, +=)
                default: 
                    #pragma omp critical
                    {
                        double old = get_value_as_double(data, data_idx);
                        set_tensor_value_from_float(data, data_idx, old + val);
                    }
            }
        } else if (reduction == 2) { // Mul
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, *=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, *=)
                default:
                    #pragma omp critical
                    {
                        double old = get_value_as_double(data, data_idx);
                        set_tensor_value_from_float(data, data_idx, old * val);
                    }
            }
        }
    }
}

// GroupNormalization
// Egor Izmaylov: Function `group_norm_forward` is the C backend entry point for the group norm operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
void group_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int num_groups, float epsilon) {
    if (!input || !scale || !B || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    
    // 检查能否整除
    if (C % num_groups != 0) return;
    int channels_per_group = C / num_groups;
    
    // 计算空间大小 (H * W * ...)
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    // 每个 Group 的元素数量
    size_t group_size = channels_per_group * spatial_size;
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            // 计算当前 Group 的 Mean 和 Var
            // Group 的数据范围：从 channel_start 到 channel_end
            int c_start = g * channels_per_group;
            int c_end = c_start + channels_per_group;
            
            double sum = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    sum += get_value_as_double(input, offset + i);
                }
            }
            double mean = sum / group_size;
            
            double sum_sq_diff = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double val = get_value_as_double(input, offset + i);
                    double diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
            }
            double var = sum_sq_diff / group_size;
            double inv_std = 1.0 / sqrt(var + epsilon);
            
            // 应用归一化和仿射变换
            for (int c = c_start; c < c_end; c++) {
                double s_val = get_value_as_double(scale, c);
                double b_val = get_value_as_double(B, c);

                double A = inv_std * s_val;
                double K = b_val - mean * A;
                
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double x = get_value_as_double(input, offset + i);
                    double y = x * A + K;
                    set_tensor_value_from_float(output, offset + i, y);
                }
            }
        }
    }
}

// Binarizer
// Egor Izmaylov: Function `binarizer_forward` is the C backend entry point for the binarizer operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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

// DynamicQuantizeLinear
// 仅支持映射到 uint8 ([0, 255])
// Egor Izmaylov: Function `dynamic_quantize_linear_forward` is the C backend entry point for the dynamic quantize linear operator; it validates tensor buffers, performs the numeric loop in C/OpenMP when applicable, and writes dtype-correct outputs.
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
