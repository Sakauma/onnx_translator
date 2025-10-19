#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdint.h>
#include <string.h>

/**
 * 数据类型枚举定义
 */
typedef enum {
    DTYPE_FLOAT16,   // 16位浮点数
    DTYPE_BFLOAT16,  // 16位bfloat格式
    DTYPE_FLOAT32,   // 32位浮点数
    DTYPE_FLOAT64,   // 64位浮点数
    DTYPE_INT8,      // 8位整数
    DTYPE_INT16,     // 16位整数
    DTYPE_INT32,     // 32位整数
    DTYPE_INT64      // 64位整数
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
 * 余弦函数前向传播
 * 
 * @param input 输入张量
 * @param output 输出张量
 */
void cos_forward(const Tensor* input, Tensor* output);


void add_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output);

void mul_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output);

void sub_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output);

void div_forward(const Tensor* input_a, const Tensor* input_b, Tensor* output);

void reshape_forward(const Tensor* input, const Tensor* new_shape, Tensor* output);
#endif
