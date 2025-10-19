// tensor_ops/tensor_ops.c
#include "tensor_ops.h"
#include <stdlib.h>
#include <math.h>

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
    // 确保余弦查找表已经初始化
    if (!cos_lut_initialized) {
        init_cos_lut();
    }

    // 遍历张量中的每个元素
    for (size_t i = 0; i < input->size; i++) {
        // 根据数据类型进行余弦计算
        switch (input->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的余弦计算
                float val = ((float*)input->data)[i];
                ((float*)output->data)[i] = (float)cos_lut_lookup((double)val);
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的余弦计算
                double val = ((double*)input->data)[i];
                ((double*)output->data)[i] = cos_lut_lookup(val);
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的余弦计算
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = float16_to_float(val);
                float result = (float)cos_lut_lookup((double)fval);
                ((uint16_t*)output->data)[i] = float_to_float16(result);
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的余弦计算
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = bfloat16_to_float(val);
                float result = (float)cos_lut_lookup((double)fval);
                ((uint16_t*)output->data)[i] = float_to_bfloat16(result);
                break;
            }
            // 整数类型的余弦函数通常没有明确定义，这里我们将其转换为浮点数处理
            // 注意：这可能会导致精度损失
            case DTYPE_INT8: {
                int8_t val = ((int8_t*)input->data)[i];
                ((int8_t*)output->data)[i] = (int8_t)round(cos_lut_lookup((double)val));
                break;
            }
            case DTYPE_INT16: {
                int16_t val = ((int16_t*)input->data)[i];
                ((int16_t*)output->data)[i] = (int16_t)round(cos_lut_lookup((double)val));
                break;
            }
            case DTYPE_INT32: {
                int32_t val = ((int32_t*)input->data)[i];
                ((int32_t*)output->data)[i] = (int32_t)round(cos_lut_lookup((double)val));
                break;
            }
            case DTYPE_INT64: {
                int64_t val = ((int64_t*)input->data)[i];
                ((int64_t*)output->data)[i] = (int64_t)round(cos_lut_lookup((double)val));
                break;
            }
        }
    }
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
        case DTYPE_INT8:
            elem_size = 1;  // 8位整数
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
    tensor->data = malloc(tensor->size * elem_size);
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


/**
 * ReLU激活函数前向传播实现
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void relu_forward(const Tensor* input, Tensor* output) {
    // 遍历张量中的每个元素
    for (size_t i = 0; i < input->size; i++) {
        // 根据数据类型进行ReLU计算
        switch (input->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的ReLU计算
                float val = ((float*)input->data)[i];
                ((float*)output->data)[i] = val > 0.0f ? val : 0.0f;
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的ReLU计算
                double val = ((double*)input->data)[i];
                ((double*)output->data)[i] = val > 0.0 ? val : 0.0;
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的ReLU计算
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = float16_to_float(val);
                ((uint16_t*)output->data)[i] = fval > 0.0f ? val : 0;
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的ReLU计算
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = bfloat16_to_float(val);
                ((uint16_t*)output->data)[i] = fval > 0.0f ? val : 0;
                break;
            }
            case DTYPE_INT8: {
                // 8位整数的ReLU计算
                int8_t val = ((int8_t*)input->data)[i];
                ((int8_t*)output->data)[i] = val > 0 ? val : 0;
                break;
            }
            case DTYPE_INT16: {
                // 16位整数的ReLU计算
                int16_t val = ((int16_t*)input->data)[i];
                ((int16_t*)output->data)[i] = val > 0 ? val : 0;
                break;
            }
            case DTYPE_INT32: {
                // 32位整数的ReLU计算
                int32_t val = ((int32_t*)input->data)[i];
                ((int32_t*)output->data)[i] = val > 0 ? val : 0;
                break;
            }
            case DTYPE_INT64: {
                // 64位整数的ReLU计算
                int64_t val = ((int64_t*)input->data)[i];
                ((int64_t*)output->data)[i] = val > 0 ? val : 0;
                break;
            }
        }
    }
}

void add_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output){
    for (size_t i = 0; i < input_a->size; i++) {
        // 根据数据类型进行add计算
        switch (input_a->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的add计算
                float val_a = ((float*)input_a->data)[i];
                float val_b = ((float*)input_b->data)[i];
                ((float*)output->data)[i] = val_a + val_b;
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的add计算
                double val_a = ((double*)input_a->data)[i];
                double val_b = ((double*)input_b->data)[i];
                ((double*)output->data)[i] = val_a + val_b;
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的add计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_float16(fval_a + fval_b);
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的add计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_bfloat16(fval_a + fval_b);
                break;
            }
            case DTYPE_INT8: {
                // 8位整数的ReLU计算
                int8_t val_a = ((int8_t*)input_a->data)[i];
                int8_t val_b = ((int8_t*)input_b->data)[i];
                ((int8_t*)output->data)[i] = val_a + val_b;
                break;
            }
            case DTYPE_INT16: {
                // 16位整数的add计算
                int16_t val_a = ((int16_t*)input_a->data)[i];
                int16_t val_b = ((int16_t*)input_b->data)[i];
                ((int16_t*)output->data)[i] = val_a + val_b;
                break;
            }
            case DTYPE_INT32: {
                // 32位整数的add计算
                int32_t val_a = ((int32_t*)input_a->data)[i];
                int32_t val_b = ((int32_t*)input_b->data)[i];
                ((int32_t*)output->data)[i] = val_a + val_b;
                break;
            }
            case DTYPE_INT64: {
                // 64位整数的add计算
                int64_t val_a = ((int64_t*)input_a->data)[i];
                int64_t val_b = ((int64_t*)input_b->data)[i];
                ((int64_t*)output->data)[i] = val_a + val_b;
                break;
            }
        }
    }
}


void mul_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output){
    for (size_t i = 0; i < input_a->size; i++) {
        // 根据数据类型进行mul计算
        switch (input_a->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的mul计算
                float val_a = ((float*)input_a->data)[i];
                float val_b = ((float*)input_b->data)[i];
                ((float*)output->data)[i] = val_a * val_b;
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的mul计算
                double val_a = ((double*)input_a->data)[i];
                double val_b = ((double*)input_b->data)[i];
                ((double*)output->data)[i] = val_a * val_b;
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的mul计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_float16(fval_a * fval_b);
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的mul计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_bfloat16(fval_a * fval_b);
                break;
            }
            case DTYPE_INT8: {
                // 8位整数的mul计算
                int8_t val_a = ((int8_t*)input_a->data)[i];
                int8_t val_b = ((int8_t*)input_b->data)[i];
                ((int8_t*)output->data)[i] = val_a * val_b;
                break;
            }
            case DTYPE_INT16: {
                // 16位整数的mul计算
                int16_t val_a = ((int16_t*)input_a->data)[i];
                int16_t val_b = ((int16_t*)input_b->data)[i];
                ((int16_t*)output->data)[i] = val_a * val_b;
                break;
            }
            case DTYPE_INT32: {
                // 32位整数的mul计算
                int32_t val_a = ((int32_t*)input_a->data)[i];
                int32_t val_b = ((int32_t*)input_b->data)[i];
                ((int32_t*)output->data)[i] = val_a * val_b;
                break;
            }
            case DTYPE_INT64: {
                // 64位整数的mul计算
                int64_t val_a = ((int64_t*)input_a->data)[i];
                int64_t val_b = ((int64_t*)input_b->data)[i];
                ((int64_t*)output->data)[i] = val_a * val_b;
                break;
            }
        }
    }
}

void sub_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output){
    for (size_t i = 0; i < input_a->size; i++) {
        // 根据数据类型进行add计算
        switch (input_a->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的add计算
                float val_a = ((float*)input_a->data)[i];
                float val_b = ((float*)input_b->data)[i];
                ((float*)output->data)[i] = val_a - val_b;
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的add计算
                double val_a = ((double*)input_a->data)[i];
                double val_b = ((double*)input_b->data)[i];
                ((double*)output->data)[i] = val_a - val_b;
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的add计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_float16(fval_a - fval_b);
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的add计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                ((uint16_t*)output->data)[i] = float_to_bfloat16(fval_a - fval_b);
                break;
            }
            case DTYPE_INT8: {
                // 8位整数的ReLU计算
                int8_t val_a = ((int8_t*)input_a->data)[i];
                int8_t val_b = ((int8_t*)input_b->data)[i];
                ((int8_t*)output->data)[i] = val_a - val_b;
                break;
            }
            case DTYPE_INT16: {
                // 16位整数的add计算
                int16_t val_a = ((int16_t*)input_a->data)[i];
                int16_t val_b = ((int16_t*)input_b->data)[i];
                ((int16_t*)output->data)[i] = val_a - val_b;
                break;
            }
            case DTYPE_INT32: {
                // 32位整数的add计算
                int32_t val_a = ((int32_t*)input_a->data)[i];
                int32_t val_b = ((int32_t*)input_b->data)[i];
                ((int32_t*)output->data)[i] = val_a - val_b;
                break;
            }
            case DTYPE_INT64: {
                // 64位整数的add计算
                int64_t val_a = ((int64_t*)input_a->data)[i];
                int64_t val_b = ((int64_t*)input_b->data)[i];
                ((int64_t*)output->data)[i] = val_a - val_b;
                break;
            }
        }
    }
}

void div_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output) {
    for (size_t i = 0; i < input_a->size; i++) {
        // 根据数据类型进行div计算
        switch (input_a->dtype) {
            case DTYPE_FLOAT32: {
                // 32位浮点数的div计算
                float val_a = ((float*)input_a->data)[i];
                float val_b = ((float*)input_b->data)[i];
                // 防止除零错误
                if (val_b == 0.0f) {
                    ((float*)output->data)[i] = 0.0f; // 或者可以设置为INFINITY
                } else {
                    ((float*)output->data)[i] = val_a / val_b;
                }
                break;
            }
            case DTYPE_FLOAT64: {
                // 64位浮点数的div计算
                double val_a = ((double*)input_a->data)[i];
                double val_b = ((double*)input_b->data)[i];
                if (val_b == 0.0) {
                    ((double*)output->data)[i] = 0.0;
                } else {
                    ((double*)output->data)[i] = val_a / val_b;
                }
                break;
            }
            case DTYPE_FLOAT16: {
                // 16位浮点数的div计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                if (fval_b == 0.0f) {
                    ((uint16_t*)output->data)[i] = float_to_float16(0.0f);
                } else {
                    ((uint16_t*)output->data)[i] = float_to_float16(fval_a / fval_b);
                }
                break;
            }
            case DTYPE_BFLOAT16: {
                // bfloat16格式的div计算
                uint16_t val_a = ((uint16_t*)input_a->data)[i];
                uint16_t val_b = ((uint16_t*)input_b->data)[i];
                float fval_a = float16_to_float(val_a);
                float fval_b = float16_to_float(val_b);
                if (fval_b == 0.0f) {
                    ((uint16_t*)output->data)[i] = float_to_bfloat16(0.0f);
                } else {
                    ((uint16_t*)output->data)[i] = float_to_bfloat16(fval_a / fval_b);
                }
                break;
            }
            case DTYPE_INT8: {
                // 8位整数的div计算
                int8_t val_a = ((int8_t*)input_a->data)[i];
                int8_t val_b = ((int8_t*)input_b->data)[i];
                if (val_b == 0) {
                    ((int8_t*)output->data)[i] = 0;
                } else {
                    ((int8_t*)output->data)[i] = val_a / val_b; // 整数除法会截断
                }
                break;
            }
            case DTYPE_INT16: {
                // 16位整数的div计算
                int16_t val_a = ((int16_t*)input_a->data)[i];
                int16_t val_b = ((int16_t*)input_b->data)[i];
                if (val_b == 0) {
                    ((int16_t*)output->data)[i] = 0;
                } else {
                    ((int16_t*)output->data)[i] = val_a / val_b;
                }
                break;
            }
            case DTYPE_INT32: {
                // 32位整数的div计算
                int32_t val_a = ((int32_t*)input_a->data)[i];
                int32_t val_b = ((int32_t*)input_b->data)[i];
                if (val_b == 0) {
                    ((int32_t*)output->data)[i] = 0;
                } else {
                    ((int32_t*)output->data)[i] = val_a / val_b;
                }
                break;
            }
            case DTYPE_INT64: {
                // 64位整数的div计算
                int64_t val_a = ((int64_t*)input_a->data)[i];
                int64_t val_b = ((int64_t*)input_b->data)[i];
                if (val_b == 0) {
                    ((int64_t*)output->data)[i] = 0;
                } else {
                    ((int64_t*)output->data)[i] = val_a / val_b;
                }
                break;
            }
        }
    }
}

void reshape_forward(const Tensor* input, const Tensor* new_shape, Tensor* output) {
    // 参数校验
    if (!input || !new_shape || !output) {
        return;
    }

    if (new_shape->ndim != 1 || new_shape->dtype != DTYPE_INT64) {
        return;
    }

    int new_ndim = new_shape->shape[0];
    if (new_ndim <= 0) {
        return;
    }

    const int64_t* shape_data = (const int64_t*)new_shape->data;
    int64_t* target_shape = (int64_t*)malloc(new_ndim * sizeof(int64_t));
    if (!target_shape) {
        return;
    }

    // 处理目标形状
    int minus_one_count = 0;
    int64_t other_dims_product = 1;
    for (int i = 0; i < new_ndim; i++) {
        int64_t dim = shape_data[i];
        if (dim == -1) {
            minus_one_count++;
            target_shape[i] = -1;
        } else if (dim <= 0) {
            free(target_shape);
            return;
        } else {
            target_shape[i] = dim;
            other_dims_product *= dim;
        }
    }

    if (minus_one_count > 1) {
        free(target_shape);
        return;
    }

    int64_t input_total = input->size;

    // 处理-1通配符
    if (minus_one_count == 1) {
        if (other_dims_product == 0 || input_total % other_dims_product != 0) {
            free(target_shape);
            return;
        }
        int64_t inferred_dim = input_total / other_dims_product;
        for (int i = 0; i < new_ndim; i++) {
            if (target_shape[i] == -1) {
                target_shape[i] = inferred_dim;
                break;
            }
        }
    }

    // 验证元素总数匹配
    int64_t output_total = 1;
    for (int i = 0; i < new_ndim; i++) {
        output_total *= target_shape[i];
    }
    if (input_total != output_total) {
        free(target_shape);
        return;
    }

    // 计算元素大小
    size_t elem_size = 0;
    switch (input->dtype) {
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
        case DTYPE_INT16:
            elem_size = 2;
            break;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
            elem_size = 4;
            break;
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
            elem_size = 8;
            break;
        case DTYPE_INT8:
            elem_size = 1;
            break;
    }

    // 处理零元素情况
    if (input_total == 0) {
        elem_size = 0;  // 避免malloc(0)的未定义行为
    }

    void* new_data = malloc(input_total * elem_size);
    if (!new_data && input_total > 0) {  // 只有非零分配失败才报错
        free(target_shape);
        return;
    }

    // 拷贝数据
    if (input_total > 0) {
        memcpy(new_data, input->data, input_total * elem_size);
    }

    // 释放原有资源
    if (output->data) {
        free(output->data);
    }
    if (output->shape) {
        free(output->shape);
    }

    // 设置输出属性
    output->data = new_data;
    output->ndim = new_ndim;
    output->shape = target_shape;
    output->dtype = input->dtype;
    output->size = input_total;
}





