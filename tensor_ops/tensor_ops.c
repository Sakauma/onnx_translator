// tensor_ops/tensor_ops.c
#include "tensor_ops.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// 余弦查找表大小
#define COS_LUT_SIZE 4096
// 余弦查找表位数
#define COS_LUT_BITS 12
// 余弦查找表
static double cos_lut[COS_LUT_SIZE + 1];
// 余弦查找表初始化标志
static int cos_lut_initialized = 0;
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
    // 提取指数位并调整偏移
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    // 提取尾数位
    uint16_t frac = (bits >> 13) & 0x3FF;
    
    // 处理特殊情况：指数过小
    if (exp <= 0) return sign;
    // 处理特殊情况：指数过大
    if (exp >= 31) return sign | 0x7C00;
    
    // 正常情况：组合符号位、指数位和尾数位
    return sign | (exp << 10) | frac;
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
    
    // 处理特殊情况：指数为0
    if (exp == 0) return *(float*)&sign;
    
    // 处理特殊情况：指数为31（无穷大或NaN）
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (frac << 13);
        return *(float*)&bits;
    }
    // 调整指数偏移
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
    // 将16位数据左移16位，低位补0
    uint32_t bits = ((uint32_t)value) << 16;
    return *(float*)&bits;
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
    
    // 设置维度数
    tensor->ndim = ndim;
    
    // 分配并复制形状数组
    tensor->shape = (int*)malloc(ndim * sizeof(int));
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
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            elem_size = 2;  // 16位数据类型
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
    }
    
    // 分配数据内存
    //tensor->data = malloc(tensor->size * elem_size);
    tensor->data = calloc(tensor->size, elem_size);
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
        case DTYPE_FLOAT16: return float16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_BFLOAT16: return bfloat16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_FLOAT32: return ((float*)tensor->data)[index];
        case DTYPE_FLOAT64: return (float)((double*)tensor->data)[index];
        case DTYPE_INT4: return (float)((int8_t*)tensor->data)[index];
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
        case DTYPE_FLOAT32: return (double)((float*)tensor->data)[index];
        case DTYPE_FLOAT16: return (double)float16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_BFLOAT16: return (double)bfloat16_to_float(((uint16_t*)tensor->data)[index]);
        case DTYPE_INT4: return (double)((int8_t*)tensor->data)[index];
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
        case DTYPE_INT4: return (int64_t)((int8_t*)tensor->data)[index];
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
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = (double)value; break;
        default: break;
    }
}

static inline void set_tensor_value_from_float(Tensor* tensor, size_t index, double value) {
    switch (tensor->dtype) {
        case DTYPE_FLOAT32: ((float*)tensor->data)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double*)tensor->data)[index] = value; break;
        // 如果目标是整数，进行截断 (Round or Cast)
        case DTYPE_INT4:    ((int8_t*)tensor->data)[index] = (int8_t)value; break; 
        case DTYPE_INT8:    ((int8_t*)tensor->data)[index] = (int8_t)value; break;
        case DTYPE_UINT8: ((uint8_t*)tensor->data)[index] = (uint8_t)value; break;
        case DTYPE_INT32:   ((int32_t*)tensor->data)[index] = (int32_t)value; break;
        case DTYPE_INT64:   ((int64_t*)tensor->data)[index] = (int64_t)value; break;
        default: break;
    }
}

/* 判断是否为整数类型 */
#define IS_INT_TYPE(d) (d == DTYPE_INT8 || d == DTYPE_INT16 || d == DTYPE_INT32 || d == DTYPE_INT64 || d == DTYPE_INT4)

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
static inline int64_t op_div(int64_t a, int64_t b) { return b == 0 ? 0 : a / b; }

/**
 * ReLU激活函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void relu_forward(const Tensor* input, Tensor* output) {
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
    for (size_t i = 0; i < input->size; i++) {
        if (IS_INT_TYPE(input->dtype)) {
            // 整数路径
            int64_t val = get_value_as_int64(input, i);
            // TODO: int64_min 的 abs 可能会溢出
            int64_t res = val < 0 ? -val : val;
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
    // 如果已经初始化，则直接返回
    if (cos_lut_initialized) return;
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
    // 处理负角度
    double reduced = x;
    if (reduced < 0) {
        reduced = -reduced;
    }
    // 将角度归一化到[0, 2π]区间
    if (reduced > TWO_PI) {
        int n = (int)(reduced / TWO_PI);
        reduced -= n * TWO_PI;
    }
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
    if (!cos_lut_initialized) init_cos_lut();

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
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_add);
    } else {
        // 浮点路径 (保持原有的 switch 结构或简化)
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) + get_value_as_double(B, i);
        } else { // 默认 float32
            float* out_data = (float*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_float(A, i) + get_value_as_float(B, i);
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
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_sub);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) - get_value_as_double(B, i);
        } else {
            float* out_data = (float*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_float(A, i) - get_value_as_float(B, i);
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
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_mul);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) * get_value_as_double(B, i);
        } else {
            float* out_data = (float*)O->data;  
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_float(A, i) * get_value_as_float(B, i);
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
    if (IS_INT_TYPE(O->dtype)) {
        BINARY_OP_INT_LOGIC(op_div);
    } else {
        if (O->dtype == DTYPE_FLOAT64) {
            double* out_data = (double*)O->data;
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_double(A, i) / get_value_as_double(B, i);
        } else {
            float* out_data = (float*)O->data;  
            #pragma omp parallel for
            for (size_t i = 0; i < O->size; i++) 
                out_data[i] = get_value_as_float(A, i) / get_value_as_float(B, i);
        }
    }
}