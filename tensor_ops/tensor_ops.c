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

// 获取数据类型的字节大小
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
static inline int8_t saturate_cast_int4(int64_t val) {
    if (val > 7) return 7;
    if (val < -8) return -8;
    return (int8_t)val;
}

// 8-bit 饱和截断
static inline int8_t saturate_cast_int8(int64_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// 8-bit 无符号饱和截断 (0 ~ 255)
static inline uint8_t saturate_cast_uint8(int64_t val) {
    if (val > 255) return 255;
    if (val < 0) return 0;
    return (uint8_t)val;
}

// 16-bit 饱和截断
static inline int16_t saturate_cast_int16(int64_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// 32-bit 饱和截断
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
static inline uint16_t float_to_float16(float value) {
    uint32_t bits = *(uint32_t*)&value;
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
static inline float float16_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000) << 16;
    uint32_t exp  = (value >> 10) & 0x1F;
    uint32_t frac = value & 0x3FF;

    if (exp == 0 && frac == 0) {
        return *(float*)&sign; 
    }
    if (exp == 31) {
        return *(float*)&(uint32_t){sign | 0x7F800000 | (frac << 13)};
    }
    if (exp == 0) {
        int32_t new_exp = -14 + 127; 
        while ((frac & 0x400) == 0) { 
            frac <<= 1;
            new_exp--;
        }
        frac &= 0x3FF; 
        uint32_t bits = sign | (new_exp << 23) | (frac << 13);
        return *(float*)&bits;
    }
    uint32_t new_exp = exp - 15 + 127;
    uint32_t bits = sign | (new_exp << 23) | (frac << 13);
    return *(float*)&bits;
}

/**
 * 将32位浮点数转换为16位bfloat16格式
 * 
 * @param value 32位浮点数
 * @return 16位bfloat16格式数据
 */
static inline uint16_t float_to_bfloat16(float value) {
    uint32_t bits = *(uint32_t*)&value;

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
static inline float bfloat16_to_float(uint16_t value) {
    // 提取符号位
    uint32_t sign = (value & 0x8000) << 16;
    // 提取指数位
    uint32_t exp = (value & 0x7F80) << 16;
    // 提取尾数位
    uint32_t frac = (value & 0x007F) << 16;
    // 组合符号位、指数位和尾数位
    uint32_t bits = sign | exp | frac;
    return *(float*)&bits;
}

/**
 * 将8位float8_e4m3格式数据转换为32位浮点数
 * 
 * @param value 8位float8_e4m3格式数据
 * @return 32位浮点数
 */
static inline float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp  = (val & 0x78) >> 3;
    uint32_t mant = (val & 0x07);
    if (exp == 0 && mant == 0) return *(float*)&sign;
    if (exp == 15 && mant == 7) {
        return *(float*)&(uint32_t){sign | 0x7F800000 | 0x400000};
    }

    if (exp == 0) {
        int32_t new_exp = -6 + 127; 
        while ((mant & 0x08) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x07;
        return *(float*)&(uint32_t){sign | (new_exp << 23) | (mant << 20)};
    }
    uint32_t new_exp = exp + 120;
    return *(float*)&(uint32_t){sign | (new_exp << 23) | (mant << 20)};
}

static inline uint8_t float_to_fp8_e4m3(float f) {
    uint32_t bits = *(uint32_t*)&f;
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
static inline float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp  = (val & 0x7C) >> 2;
    uint32_t mant = (val & 0x03);
    if (exp == 0 && mant == 0) return *(float*)&sign;
    if (exp == 31) {
        uint32_t f32_mant = mant << 21;
        if (mant != 0) f32_mant |= 0x400000; 
        return *(float*)&(uint32_t){sign | 0x7F800000 | f32_mant};
    }
    if (exp == 0) {
        int32_t new_exp = -14 + 127;
        while ((mant & 0x04) == 0) {
            mant <<= 1;
            new_exp--;
        }
        mant &= 0x03;
        return *(float*)&(uint32_t){sign | (new_exp << 23) | (mant << 21)};
    }
    uint32_t new_exp = exp + 112;
    return *(float*)&(uint32_t){sign | (new_exp << 23) | (mant << 21)};
}

static inline uint8_t float_to_fp8_e5m2(float f) {
    uint32_t bits = *(uint32_t*)&f;
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
Tensor* create_tensor(int* shape, int ndim, DataType dtype) {
    // 分配张量结构体内存
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        return NULL;
    }
    
    // 设置维度数
    tensor->ndim = ndim;
    
    // 分配并复制形状数组
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    
    // 设置数据类型
    tensor->dtype = dtype;
    
    // 计算总元素数
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
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
    tensor->data = calloc(tensor->size, elem_size);
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

/* 判断是否为整数类型 */
#define IS_INT_TYPE(d) (d == DTYPE_INT8 || d == DTYPE_UINT8 || d == DTYPE_INT16 || d == DTYPE_INT32 || d == DTYPE_INT64 || d == DTYPE_INT4)

// --- 通用一元算子宏模板 ---
#ifndef UNARY_OP_IMPL
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
static inline int64_t op_add(int64_t a, int64_t b) { return a + b; }
static inline int64_t op_sub(int64_t a, int64_t b) { return a - b; }
static inline int64_t op_mul(int64_t a, int64_t b) { return a * b; }
static inline int64_t op_div(int64_t a, int64_t b) { return b == 0 ? (a >= 0 ? INT64_MAX : INT64_MIN) : a / b; }

// 安全获取4D张量的值
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
            int g = m / (out_c / group);
            
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
void flatten_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    size_t total_bytes = input->size * elem_size;
    memcpy(output->data, input->data, total_bytes);
}

// Reshape 实现
void reshape_forward(const Tensor* input, Tensor* output) {
    flatten_forward(input, output);
}

// 从平坦索引反解 N 维坐标
static inline void get_coords_from_index(size_t index, int* coords, int* shape, int ndim) {
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = index % shape[i];
        index /= shape[i];
    }
}

// 从 N 维坐标计算平坦索引
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
void transpose_forward(const Tensor* input, Tensor* output, int* perm) {
    if (!input || !output || !perm) return;
    int ndim = input->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM]; // 输出坐标
        int in_coords[MAX_NDIM];  // 输入坐标
        
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
static inline int64_t op_max(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t op_min(int64_t a, int64_t b) { return a < b ? a : b; }

// Pow 实现
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
            double val = get_value_as_double(src, src_idx);
            set_tensor_value_from_float(output, i, val);
        }
    }
}

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
        double val = get_value_as_double(input, in_idx);
        set_tensor_value_from_float(output, i, val);
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
void cast_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) return;
    
    // 检查是否是 "浮点 -> 整数" 的情况
    int is_float_to_int = (input->dtype == DTYPE_FLOAT32 || input->dtype == DTYPE_FLOAT64 || 
                           input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16) &&
                          IS_INT_TYPE(output->dtype);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        // 1. 读取输入 (统一转 double)
        double val = get_value_as_double(input, i);
        
        // 2. 写入输出
        if (is_float_to_int) {
            int64_t trunc_val = (int64_t)val;
            set_tensor_value_from_int(output, i, trunc_val);
        } else {
            // 其他情况 (Int->Float, Float->Float, Int->Int) 保持原有逻辑
            set_tensor_value_from_float(output, i, val);
        }
    }
}

// Clip：支持全广播
// 调用此函数前，Python 端已将 input, min_t, max_t 广播为相同形状
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
void matmul_forward(const Tensor* A, const Tensor* B, Tensor* Y) {
    if (!A || !B || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }
    if (ndim < 2) return; // 至少是 2D
    int M = A->shape[A->ndim - 2];
    int K = A->shape[A->ndim - 1];
    int N = B->shape[B->ndim - 1];
    #pragma omp parallel for
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM]; // 最大 16 维
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

// Gather 实现
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
        double val = get_value_as_double(input, in_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// Shape 实现
void shape_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    // Output 应该是一个 1D int64 张量，长度等于 input->ndim
    int64_t* out_data = (int64_t*)output->data;
    for (int i = 0; i < input->ndim; i++) {
        out_data[i] = (int64_t)input->shape[i];
    }
}

// 比较 A 和 B，结果存入 O (通常是 uint8)
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

void identity_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
}

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

void where_forward(const Tensor* Cond, const Tensor* X, const Tensor* Y, Tensor* O) {
    if (!Cond || !X || !Y || !O) return;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < O->size; i++) {
        double c_val = get_value_as_double(Cond, i);
        double x_val = get_value_as_double(X, i);
        double y_val = get_value_as_double(Y, i);
        
        // 条件非 0 即为 True
        double res = (c_val != 0) ? x_val : y_val;
        set_tensor_value_from_float(O, i, res);
    }
}

// ConstantOfShape
void constant_of_shape_forward(Tensor* output, const Tensor* value) {
    if (!output) return;

    double fill_val = 0.0;
    if (value && value->data) {
        fill_val = get_value_as_double(value, 0);
    }

    size_t loop_size = output->size;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        set_tensor_value_from_float(output, i, fill_val);
    }
}

// Range
void range_forward(const Tensor* start, const Tensor* limit, const Tensor* delta, Tensor* output) {
    if (!start || !delta || !output) return;
    
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
void tile_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim = input->ndim;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        for (int d = 0; d < ndim; d++) {
            in_coords[d] = out_coords[d] % input->shape[d];
        }

        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        double val = get_value_as_double(input, in_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// Pad
// mode: 0=constant, 1=reflect, 2=edge
void pad_forward(const Tensor* data, Tensor* output, const Tensor* pads, const Tensor* constant_value, int mode) {
    if (!data || !output || !pads) return;
    
    int ndim = data->ndim;
    
    int64_t pad_begins[MAX_NDIM];
    int64_t pad_ends[MAX_NDIM];
    for (int d = 0; d < ndim; d++) {
        pad_begins[d] = get_value_as_int64(pads, d);
        pad_ends[d]   = get_value_as_int64(pads, d + ndim);
    }
    
    double const_val = 0.0;
    if (constant_value && constant_value->data) {
        const_val = get_value_as_double(constant_value, 0);
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
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
                }
            }
        }
        
        if (in_bounds) {
            size_t in_idx = get_index_from_coords(in_coords, data->shape, ndim);
            double val = get_value_as_double(data, in_idx);
            set_tensor_value_from_float(output, i, val);
        } else {
            set_tensor_value_from_float(output, i, const_val);
        }
    }
}

// 检查某个轴是否在归约列表中
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

#define ARG_OP_IMPL(FUNC_NAME, INIT_VAL, CMP_OP) \
void FUNC_NAME(const Tensor* input, Tensor* output, int axis, int select_last_index) { \
    if (!input || !output) return; \
    int ndim = input->ndim; \
    int axis_dim = input->shape[axis]; \
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
            else coords[d] = out_coords[out_ptr++]; \
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
void gather_elements_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;
    
    int ndim = data->ndim;
    if (axis < 0) axis += ndim;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
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

void topk_forward(const Tensor* input, Tensor* values, Tensor* indices, int axis, int largest, int sorted, int K) {
    if (!input || !values || !indices) return;
    
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

static uint32_t simple_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

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
static inline int64_t op_bitwise_and(int64_t a, int64_t b) { return a & b; }
static inline int64_t op_bitwise_or(int64_t a, int64_t b) { return a | b; }
static inline int64_t op_bitwise_xor(int64_t a, int64_t b) { return a ^ b; }
static inline int64_t op_shift_left(int64_t a, int64_t b) { return a << b; }
static inline int64_t op_shift_right(int64_t a, int64_t b) { return a >> b; }

// BitwiseAnd
void bitwise_and_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_and); 
}

// BitwiseOr
void bitwise_or_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_or);
}

// BitwiseXor
void bitwise_xor_forward(const Tensor* A, const Tensor* B, Tensor* O) {
    if (!A || !B || !O) return;
    BINARY_OP_INT_LOGIC(op_bitwise_xor);
}

// BitwiseNot
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
void global_average_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    if (ndim != 4) {
        return;
    }
    
    // N = 前 ndim-2 维的乘积
    int N = 1;
    for (int i = 0; i < ndim - 2; i++) N *= input->shape[i];
    int H = input->shape[ndim - 2];
    int W = input->shape[ndim - 1];
    int spatial_size = H * W;
    
    _Pragma("omp parallel for")
    for (int n = 0; n < N; n++) {
        double sum = 0.0;
        size_t offset = (size_t)n * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            sum += get_value_as_double(input, offset + i);
        }
        set_tensor_value_from_float(output, n, sum / spatial_size);
    }
}

// GlobalMaxPool
void global_max_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    int N = 1;
    for (int i = 0; i < ndim - 2; i++) N *= input->shape[i];
    int H = input->shape[ndim - 2];
    int W = input->shape[ndim - 1];
    int spatial_size = H * W;
    
    _Pragma("omp parallel for")
    for (int n = 0; n < N; n++) {
        double max_val = -DBL_MAX;
        size_t offset = (size_t)n * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            if (val > max_val) max_val = val;
        }
        set_tensor_value_from_float(output, n, max_val);
    }
}

// GlobalLpPool
void global_lp_pool_forward(const Tensor* input, Tensor* output, int p) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    // N = 前 ndim-2 维的乘积 (Batch * Channel)
    int N = 1;
    for (int i = 0; i < ndim - 2; i++) N *= input->shape[i];
    int H = input->shape[ndim - 2];
    int W = input->shape[ndim - 1];
    int spatial_size = H * W;
    
    _Pragma("omp parallel for")
    for (int n = 0; n < N; n++) {
        double sum_pow = 0.0;
        size_t offset = (size_t)n * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            sum_pow += pow(fabs(val), p);
        }
        
        // p=1 时就是 Sum(|x|)，p=2 时是 L2 Norm，p=inf 时是 Max
        double res = pow(sum_pow, 1.0 / p);
        set_tensor_value_from_float(output, n, res);
    }
}

// Mean (Element-wise)
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

void size_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int64_t total_elems = (int64_t)input->size;
    set_tensor_value_from_int(output, 0, total_elems);
}

// IsInf
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
void one_hot_forward(const Tensor* indices, const Tensor* values, Tensor* output, int axis) {
    if (!indices || !values || !output) return;
    
    int out_ndim = output->ndim;
    if (axis < 0) axis += out_ndim;
    
    int depth = output->shape[axis];

    double off_val = get_value_as_double(values, 0);
    double on_val = get_value_as_double(values, 1);
    
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
        
        double res = (current_depth_idx == target_idx) ? on_val : off_val;
        set_tensor_value_from_float(output, i, res);
    }
}

// Tril / Triu
void triangular_forward(const Tensor* input, Tensor* output, int k, int upper) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return; 
    
    int H = input->shape[ndim - 2];
    int W = input->shape[ndim - 1];
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        int coords[MAX_NDIM];
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
static int64_t get_window_size(const Tensor* size_tensor) {
    if (!size_tensor) return 0;
    return get_value_as_int64(size_tensor, 0);
}

// Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
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
            // Avoid division by zero
            if (norm < 1e-12) norm = 1e-12; 
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                set_tensor_value_from_float(output, idx, val / norm);
            }
        }
    }
}

// DepthToSpace
void depth_to_space_forward(const Tensor* input, Tensor* output, int blocksize, int mode) {
    if (!input || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    
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
void space_to_depth_forward(const Tensor* input, Tensor* output, int blocksize) {
    if (!input || !output) return;
    
    int N = output->shape[0];
    int C_out = output->shape[1];
    int H_out = output->shape[2];
    int W_out = output->shape[3];
    
    int C_in = input->shape[1];
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            int in_c = c / (blocksize * blocksize);
            int rem = c % (blocksize * blocksize);
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
void reverse_sequence_forward(const Tensor* input, const Tensor* sequence_lens, Tensor* output, int time_axis, int batch_axis) {
    if (!input || !output || !sequence_lens) return;
    int ndim = input->ndim;
    if (time_axis < 0) time_axis += ndim;
    if (batch_axis < 0) batch_axis += ndim;
    
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
    
    int batch_dim = input->shape[batch_axis];

    size_t strides[MAX_NDIM];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) strides[i] = strides[i+1] * input->shape[i+1];
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM];
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
        int coords[MAX_NDIM];
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