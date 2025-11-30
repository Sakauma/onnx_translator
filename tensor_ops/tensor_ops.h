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

/**
 * Exp 指数函数前向传播 (y = e^x)
 */
void exp_forward(const Tensor* input, Tensor* output);

/**
 * Log 自然对数函数前向传播 (y = ln(x))
 */
void log_forward(const Tensor* input, Tensor* output);

/**
 * Sqrt 平方根函数前向传播 (y = x^0.5)
 */
void sqrt_forward(const Tensor* input, Tensor* output);

/**
 * Sigmoid 激活函数前向传播 (y = 1 / (1 + e^-x))
 */
void sigmoid_forward(const Tensor* input, Tensor* output);

/**
 * Tanh 激活函数前向传播 (y = tanh(x))
 */
void tanh_forward(const Tensor* input, Tensor* output);

/**
 * Flatten 前向传播
 * 将输入张量展平为 2D 输出 [batch, remaining]
 */
void flatten_forward(const Tensor* input, Tensor* output);

/**
 * Reshape 前向传播
 * 改变张量形状
 */
void reshape_forward(const Tensor* input, Tensor* output);

/**
 * Transpose 前向传播
 * 根据 perm 置换维度
 * input: 输入张量
 * output: 输出张量 (形状已在 Python 层计算好)
 * perm: 维度置换数组 (例如 [0, 3, 1, 2])
 */
void transpose_forward(const Tensor* input, Tensor* output, int* perm);

/**
 * Pow 幂运算 (Y = A ^ B)
 */
void pow_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * Max 最大值 (Y = max(A, B))
 */
void max_forward(const Tensor* A, const Tensor* B, Tensor* O);

/**
 * Min 最小值 (Y = min(A, B))
 */
void min_forward(const Tensor* A, const Tensor* B, Tensor* O);

/* Squeeze 和 Unsqueeze 本质是 Reshape，直接复用 reshape_forward 或 flatten_forward 即可，在 C 层不需要新函数 */

/**
 * Concat 拼接算子前向传播
 * @param inputs 输入张量指针数组
 * @param num_inputs 输入张量的数量
 * @param output 输出张量
 * @param axis 拼接的维度轴
 */
void concat_forward(const Tensor** inputs, int num_inputs, Tensor* output, int axis);

/**
 * Slice 切片算子前向传播
 * @param input 输入张量
 * @param output 输出张量
 * @param starts 起始索引数组 (长度必须等于 ndim)
 * @param steps 步长数组 (长度必须等于 ndim)
 */
void slice_forward(const Tensor* input, Tensor* output, int* starts, int* steps);

/**
 * Neg 取负 (Y = -X)
 */
void neg_forward(const Tensor* input, Tensor* output);

/**
 * Reciprocal 倒数 (Y = 1 / X)
 */
void reciprocal_forward(const Tensor* input, Tensor* output);

/**
 * Clip 数值截断
 * min_val/max_val 为标量指针，如果为 NULL 表示无下界/无上界
 */
void clip_forward(const Tensor* input, Tensor* output, const Tensor* min, const Tensor* max);

/**
 * Cast 类型转换
 * 本质上就是从 Input 读取 (自动转double) 再写入 Output (自动转目标类型)
 */
void cast_forward(const Tensor* input, Tensor* output);

/**
 * Ceil 向上取整
 */
void ceil_forward(const Tensor* input, Tensor* output);

/**
 * Floor 向下取整
 */
void floor_forward(const Tensor* input, Tensor* output);

/**
 * MatMul 矩阵乘法 (支持广播)
 * Y = A @ B
 */
void matmul_forward(const Tensor* A, const Tensor* B, Tensor* Y);

#endif
