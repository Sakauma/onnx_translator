// tensor_ops/tensor_ops.c
#include "tensor_ops.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

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
    // 获取32位浮点数的位表示
    uint32_t bits = *(uint32_t*)&value;
    // 提取符号位
    uint16_t sign = (bits >> 16) & 0x8000;
    // 提取指数位
    uint32_t exp = (bits >> 23) & 0xFF;
    // 提取尾数位
    uint32_t frac = bits & 0x7FFFFF;
    
    // 处理特殊情况：NaN
    if (exp == 0xFF && frac != 0) {
        // 返回NaN
        return sign | 0x7FFF;
    }
    // 处理特殊情况：无穷大             
    if (exp == 0xFF && frac == 0) {
        // 返回无穷大
        return sign | 0x7C00;
    }
    // 处理特殊情况：零
    if (exp == 0 && frac == 0) {
        // 返回零
        return sign;
    }
    // 处理特殊情况：次正规数
    if (exp == 0) {
        // 次正规数，调整为正规数
        int shift = __builtin_clz(frac) - 8; // 计算前导零的数量
        frac <<= shift;
        exp = 1 - shift - 127 + 15;
        frac >>= 13;
        return sign | (exp << 10) | frac;
    }
    // 正常情况：调整指数偏移
    exp = exp - 127 + 15;
    // 处理特殊情况：指数过大
    if (exp >= 31) {
        // 返回无穷大
        return sign | 0x7C00;
    }
    // 处理特殊情况：指数过小
    if (exp <= 0) {
        // 返回零
        return sign;
    }
    // 正常情况：组合符号位、指数位和尾数位
    return sign | (exp << 10) | (frac >> 13);
}

/**
 * 将16位浮点数转换为32位浮点数
 * 
 * @param value 16位浮点数
 * @return 32位浮点数
 */
static inline float float16_to_float(uint16_t value) {
    // 提取符号位并左移16位
    uint32_t sign = (value & 0x8000) << 16;
    // 提取指数位
    uint32_t exp = (value >> 10) & 0x1F;
    // 提取尾数位
    uint32_t frac = value & 0x3FF;
    
    // 处理特殊情况：零
    if (exp == 0 && frac == 0) {
        return *(float*)&sign;
    }
    // 处理特殊情况：无穷大或NaN
    if (exp == 0x1F) {
        uint32_t bits = sign | 0x7F800000 | (frac << 13);
        return *(float*)&bits;
    }
    // 处理特殊情况：次正规数
    if (exp == 0) {
        // 次正规数，调整为正规数
        int shift = 14 - __builtin_clz(frac); // 计算前导零的数量
        frac <<= shift;
        exp = 1 - shift - 15 + 127;
        frac >>= 13;
        uint32_t bits = sign | (exp << 23) | (frac << 13);
        return *(float*)&bits;
    }
    // 正常情况：调整指数偏移
    exp = exp - 15 + 127;
    // 组合符号位、指数位和尾数位
    uint32_t bits = sign | (exp << 23) | (frac << 13);
    return *(float*)&bits;
}

/**
 * 将32位浮点数转换为16位bfloat16格式
 * 
 * @param value 32位浮点数
 * @return 16位bfloat16格式数据
 */
static inline uint16_t float_to_bfloat16(float value) {
    // 获取32位浮点数的位表示
    uint32_t bits = *(uint32_t*)&value;
    // 取高16位作为bfloat16
    return (uint16_t)(bits >> 16);
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
    uint32_t sign = (val & 0x80) << 24;
    uint32_t exp = (val & 0x78) >> 3;
    uint32_t mant = (val & 0x07);

    if (exp == 0) {
        if (mant == 0) return *(float*)&sign;
        // 次正规数处理: 0.mant * 2^(-6)
        float result = (float)mant * powf(2.0f, -6.0f);
        return (val & 0x80) ? -result : result;
    } else if (exp == 0x0F && mant == 0x07) {
        return NAN; // NaN
    } else {
        // 正规数: 1.mant * 2^(exp - 7)
        // 转 float32: exp - 7 + 127 = exp + 120
        uint32_t new_exp = exp + 120;
        uint32_t new_mant = mant << 20; // 3位尾数左移填充23位
        uint32_t bits = sign | (new_exp << 23) | new_mant;
        return *(float*)&bits;
    }
}

static inline uint8_t float_to_fp8_e4m3(float f) {
    uint32_t bits = *(uint32_t*)&f;
    uint32_t sign = (bits & 0x80000000) >> 24;
    uint32_t exp = (bits & 0x7F800000) >> 23;
    uint32_t mant = (bits & 0x007FFFFF);

    // 1. NaN 处理
    if (exp == 0xFF && mant != 0) return 0x7F | sign;

    // 2. 零处理
    if (exp == 0) return (uint8_t)sign;

    // 3. 计算指数 (Bias 127 -> 7)
    int e_fp8 = (int)exp - 127 + 7;

    // 4. 下溢处理 (简化为0)
    if (e_fp8 <= 0) return (uint8_t)sign;

    // 5. 上溢/无穷大处理 (饱和到最大值 448 = 0x7E)
    // E4M3 没有 Inf，通常饱和处理
    if (e_fp8 > 15) return 0x7E | sign;

    // 6. 尾数舍入
    // 取高3位 (bit 22, 21, 20)
    uint32_t m_fp8 = (mant >> 20) & 0x7;
    // 检查第19位进行四舍五入
    if ((mant >> 19) & 1) {
        m_fp8++;
        if (m_fp8 > 7) { // 尾数进位导致溢出
            m_fp8 = 0;
            e_fp8++;
        }
    }
    
    // 再次检查指数溢出 (可能因为进位导致)
    // 注意: E4M3 最大正规数 exp=15, mant=6 (0x7E)。Exp=15, Mant=7 是 NaN
    if (e_fp8 > 15 || (e_fp8 == 15 && m_fp8 == 7)) return 0x7E | sign;

    return (uint8_t)(sign | (e_fp8 << 3) | m_fp8);
}

/**
 * 将8位float8_e5m2格式数据转换为32位浮点数
 * 
 * @param value 8位float8_e5m2格式数据
 * @return 32位浮点数
 */
static inline float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = (val & 0x80) << 24;
    uint32_t exp = (val & 0x7C) >> 2;
    uint32_t mant = (val & 0x03);

    if (exp == 0) {
        if (mant == 0) return *(float*)&sign;
        // 次正规数: 0.mant * 2^(-14)
        float result = (float)mant * powf(2.0f, -14.0f);
        return (val & 0x80) ? -result : result;
    } else if (exp == 0x1F) {
        // Inf 或 NaN
        uint32_t bits = sign | 0x7F800000 | (mant ? (1 << 22) : 0);
        return *(float*)&bits;
    } else {
        // 正规数: exp - 15 + 127 = exp + 112
        uint32_t new_exp = exp + 112;
        uint32_t new_mant = mant << 21;
        uint32_t bits = sign | (new_exp << 23) | new_mant;
        return *(float*)&bits;
    }
}

static inline uint8_t float_to_fp8_e5m2(float f) {
    uint32_t bits = *(uint32_t*)&f;
    uint32_t sign = (bits & 0x80000000) >> 24;
    uint32_t exp = (bits & 0x7F800000) >> 23;
    uint32_t mant = (bits & 0x007FFFFF);

    // 1. NaN 和 Inf 处理
    if (exp == 0xFF) {
        // 保持符号，指数全1 (0x1F)
        // 如果原 mant!=0 (NaN)，则设 m_fp8=1 (非零即可)
        return (uint8_t)(sign | 0x7C | (mant ? 1 : 0));
    }

    // 2. 零处理
    if (exp == 0) return (uint8_t)sign;

    // 3. 指数转换 (Bias 127 -> 15)
    int e_fp8 = (int)exp - 127 + 15;

    // 4. 上溢处理 (变成 Inf)
    if (e_fp8 >= 31) return (uint8_t)(sign | 0x7C);

    // 5. 下溢处理
    if (e_fp8 <= 0) return (uint8_t)sign;

    // 6. 尾数舍入
    uint32_t m_fp8 = (mant >> 21) & 0x3;
    if ((mant >> 20) & 1) {
        m_fp8++;
        if (m_fp8 > 3) {
            m_fp8 = 0;
            e_fp8++;
        }
    }
    
    if (e_fp8 >= 31) return (uint8_t)(sign | 0x7C);

    return (uint8_t)(sign | (e_fp8 << 2) | m_fp8);
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
        case DTYPE_FLOAT32: return (int64_t)roundf(((float*)tensor->data)[index]);
        case DTYPE_FLOAT16: return (int64_t)roundf(float16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_BFLOAT16: return (int64_t)roundf(bfloat16_to_float(((uint16_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E4M3: return (int64_t)roundf(fp8_e4m3_to_float(((uint8_t*)tensor->data)[index]));
        case DTYPE_FLOAT8_E5M2: return (int64_t)roundf(fp8_e5m2_to_float(((uint8_t*)tensor->data)[index]));
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
        case DTYPE_FLOAT64: return (int64_t)round(((double*)tensor->data)[index]);
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
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = (double)value; break;
        default: break;
    }
}

static inline void set_tensor_value_from_float(Tensor* tensor, size_t index, double value) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT8_E4M3: ((uint8_t*)tensor->data)[index] = float_to_fp8_e4m3((float)value); break;
        case DTYPE_FLOAT8_E5M2: ((uint8_t*)tensor->data)[index] = float_to_fp8_e5m2((float)value); break;
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = value; break;
        // 如果目标是整数，使用饱和截断转换
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = saturate_cast_int4((int64_t)round(value)); break; 
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = saturate_cast_int8((int64_t)round(value)); break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = saturate_cast_uint8((int64_t)round(value)); break;
        case DTYPE_INT16:   ((int16_t*)tensor->data)[index] = saturate_cast_int16((int64_t)round(value)); break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = saturate_cast_int32((int64_t)round(value)); break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = (int64_t)round(value); break;
        default: break;
    }
}

/* 判断是否为整数类型 */
#define IS_INT_TYPE(d) (d == DTYPE_INT8 || d == DTYPE_UINT8 || d == DTYPE_INT16 || d == DTYPE_INT32 || d == DTYPE_INT64 || d == DTYPE_INT4)

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
    // 首先检查是否已初始化，避免不必要的锁操作
    if (cos_lut_initialized) return;
    
    // 加锁保护初始化过程
    pthread_mutex_lock(&cos_lut_mutex);
    
    // 再次检查是否已初始化，防止多个线程同时等待锁
    if (cos_lut_initialized) {
        pthread_mutex_unlock(&cos_lut_mutex);
        return;
    }
    
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
    // 标记查找表已初始化
    cos_lut_initialized = 1;
    
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
                double val_b = get_value_as_double(B, i);
                if (val_b == 0.0) {
                    // 除数为零，返回NaN
                    out_data[i] = 0.0 / 0.0;
                } else {
                    out_data[i] = get_value_as_double(A, i) / val_b;
                }
            }
        } else {
            // 对所有非double浮点类型使用统一处理，包括float8
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) {
                double val_a = get_value_as_double(A, i);
                double val_b = get_value_as_double(B, i);
                double res;
                if (val_b == 0.0) {
                    // 除数为零，返回NaN
                    res = 0.0 / 0.0;
                } else {
                    res = val_a / val_b;
                }
                set_tensor_value_from_float(O, i, res);
            }
        }
    }
}