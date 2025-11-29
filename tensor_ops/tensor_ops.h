#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdint.h>
#include <string.h>

/**
 * 数据类型枚举定义
 */
typedef enum {
    DTYPE_FLOAT8_E4M3, // 8位浮点数，适合推理
    DTYPE_FLOAT8_E5M2, // 8位浮点数，适合训练
    DTYPE_FLOAT16,     // 16位浮点数
    DTYPE_BFLOAT16,    // 16位bfloat格式
    DTYPE_FLOAT32,     // 32位浮点数
    DTYPE_FLOAT64,     // 64位浮点数
    DTYPE_INT4,        // 4位整数
    DTYPE_INT8,        // 8位整数
    DTYPE_UINT8,       // 8位无符号整数
    DTYPE_INT16,       // 16位整数
    DTYPE_INT32,       // 32位整数
    DTYPE_INT64,       // 64位整数
} DataType;

/**
 * 张量结构体定义
 */
typedef struct {
    void* data;      // 数据指针
    int* shape;      // 形状数组
    int ndim;        // 维度数
    size_t size;     // 总元素数
    DataType dtype;  // 数据类型
} Tensor;

/**
 * 卷积参数结构体定义
 */
typedef struct {
    int* pads;      // [top, left, bottom, right]
    int* strides;   // [h, w]
    int* dilations; // [h, w]
    int group;
} ConvParams;

/**
 * 池化参数结构体定义
 */
typedef struct {
    int* pads;         // [top, left, bottom, right]
    int* strides;      // [h, w]
    int* kernel_shape; // [h, w]
} PoolParams;

/**
 * 创建张量
 * 
 * @param shape 张量形状数组
 * @param ndim 张量维度数
 * @param dtype 数据类型
 * @return 创建的张量指针
 */
Tensor* create_tensor(int* shape, int ndim, DataType dtype);

/**
 * 释放张量内存
 * 
 * @param tensor 要释放的张量指针
 */
void free_tensor(Tensor* tensor);

/**
 * ReLU激活函数前向传播
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void relu_forward(const Tensor* input, Tensor* output);

/**
 * 初始化余弦查找表
 * 使用泰勒级数展开计算余弦值并存储在查找表中
 */
void init_cos_lut(void);

/**
 * 余弦函数前向传播
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void cos_forward(const Tensor* input, Tensor* output);

/**
 * Abs函数前向传播
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void abs_forward(const Tensor* input, Tensor* output);

/**
 * Add函数前向传播
 * 
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量
 */
void add_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * Sub函数前向传播
 * 
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量
 */
void sub_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * Mul函数前向传播
 * 
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量
 */
void mul_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * Div函数前向传播
 *  
 * @param A 输入张量A
 * @param B 输入张量B
 * @param O 输出张量
 */
void div_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * QuantizeLinear 前向传播 (FP32 -> INT8/UINT8 等)
 * 公式: y = saturate(round(x / scale) + zero_point)
 */
void quantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y);

/**
 * DequantizeLinear 前向传播 (INT8/UINT8 -> FP32 等)
 * 公式: y = (x - zero_point) * scale
 */
void dequantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y);

/**
 * Conv2D 前向传播
 * 公式: Y = Sum(X * W) + B
 */
void conv2d_forward(const Tensor* X, const Tensor* W, const Tensor* B, Tensor* Y, ConvParams* params);
      
/**
 * MaxPool 前向传播
 */
void max_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params);

/**
 * Gemm (General Matrix Multiply) 前向传播
 * 公式: Y = alpha * A' * B' + beta * C
 * transA/transB: 0=不转置, 1=转置
 */
void gemm_forward(const Tensor* A, const Tensor* B, const Tensor* C, Tensor* Y, 
                  float alpha, float beta, int transA, int transB);

/**
 * Softmax 前向传播
 */
void softmax_forward(const Tensor* input, Tensor* output, int axis);

#endif
