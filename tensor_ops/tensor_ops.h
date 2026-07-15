/**
  ******************************************************************************
  * @file        tensor_ops.h
  * @author      Egor Izmaylov
  * @brief       声明 C 后端公共 ABI、张量结构和算子 forward 接口。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdint.h>
#include <string.h>

/**
 * 公共 ABI 约定：
 *
 * - 除 create_tensor/free_tensor 外，所有 Tensor 和参数数组均由调用方持有，函数只在
 *   调用期间借用指针，不保存地址，也不释放传入内存。
 * - 输出 Tensor 必须由调用方按最终 shape、dtype 和容量预先分配；forward 接口只写
 *   data，不调整 shape、ndim、size 或 dtype。
 * - 只有本头文件明确标注“可为 NULL”的指针才允许缺省。其余接口通常以静默返回处理
 *   无效输入，因此 Python/ctypes 边界必须先完成形状、类型和必需参数校验。
 * - 多字节元素使用宿主机本地字节序。int2/int4/uint2/uint4 当前每个逻辑元素占一个
 *   字节，未做位打包；低精度浮点同样按各自编码位存储。
 */

/**
 * Python 的 DTYPE_MAP 直接依赖这些序号。已有枚举值不得重排或插入，新类型只能追加，
 * 否则 ctypes 会用错误的元素宽度解释同一缓冲区。
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
    DTYPE_UINT16,      // 16位无符号整数
    DTYPE_UINT32,      // 32位无符号整数
    DTYPE_UINT64,      // 64位无符号整数
    DTYPE_BOOL,        // 布尔值
    DTYPE_COMPLEX64,   // 64位复数
    DTYPE_COMPLEX128,  // 128位复数
    DTYPE_FLOAT8_E4M3FNUZ, // 8位 FNUZ E4M3 浮点数
    DTYPE_FLOAT8_E5M2FNUZ, // 8位 FNUZ E5M2 浮点数
    DTYPE_UINT4,       // 4位无符号整数
    DTYPE_INT2,        // 2位整数
    DTYPE_UINT2,       // 2位无符号整数
    DTYPE_FLOAT4_E2M1, // 4位 E2M1 浮点数
    DTYPE_FLOAT8_E8M0, // 8位 E8M0 浮点数
} DataType;

/**
 * C 后端的拥有型连续张量描述符。
 *
 * create_tensor 创建的实例同时拥有 data 和 shape；size 等于各维长度之积，标量
 * (ndim == 0) 的 size 为 1。该字段布局必须与 nn.CTensor 完全一致。
 */
typedef struct {
    void* data;      // 数据指针
    int* shape;      // 形状数组
    int ndim;        // 维度数
    size_t size;     // 总元素数
    DataType dtype;  // 数据类型
} Tensor;

/**
 * 2D 卷积参数视图。三个数组均由调用方持有，并在 forward 返回前保持有效。
 */
typedef struct {
    int* pads;      // [top, left, bottom, right]
    int* strides;   // [h, w]
    int* dilations; // [h, w]
    int group;
} ConvParams;

/**
 * 2D 池化参数视图。pads 顺序为 [top, left, bottom, right]，其余数组长度为 2。
 */
typedef struct {
    int* pads;         // [top, left, bottom, right]
    int* strides;      // [h, w]
    int* dilations;    // [h, w]
    int* kernel_shape; // [h, w]
} PoolParams;

/**
 * 归约参数视图。axes 由调用方持有，长度为 num_axes；轴值按具体算子约定解释。
 */
typedef struct {
    int* axes;       // 要归约的轴数组
    int num_axes;    // 轴的数量
    int keepdims;    // 是否保持维度
} ReduceParams;

/**
 * 创建零初始化的拥有型张量，并复制 shape 数组。
 *
 * @param shape 长度为 ndim 的形状数组；仅 ndim == 0 时可为 NULL。
 * @param ndim 非负维数；任一维也必须非负。
 * @param dtype 逻辑数据类型。
 * @return 成功时返回新张量；参数非法或任一步分配失败时返回 NULL。
 */
Tensor* create_tensor(int* shape, int ndim, DataType dtype);

/**
 * 释放 create_tensor 创建的 data、shape 和描述符；传入 NULL 是安全的。
 * 调用后 tensor 及其所有字段地址均失效，不能用于释放借用或栈上构造的 Tensor。
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
 * X/Scale/ZeroPoint 必须已广播到 Y 的元素布局；即使 ONNX 省略 zero_point，调用层
 * 也要传入同目标 dtype 的零值 Tensor，不能传 NULL。
 */
void quantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y);

/**
 * QuantizeLinear 前向传播，显式指定除法精度属性
 * precision 使用 ONNX TensorProto dtype 编码；0 表示沿用默认 Scale dtype 语义
 */
void quantize_linear_forward_precision(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y, int precision);

/**
 * QuantizeLinear 前向传播，显式指定除法精度和 float8 溢出饱和属性
 * precision 使用 ONNX TensorProto dtype 编码；saturate 仅影响 float8 目标 dtype
 */
void quantize_linear_forward_precision_saturate(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y, int precision, int saturate);

/**
 * DequantizeLinear 前向传播 (INT8/UINT8 -> FP32 等)
 * 公式: y = (x - zero_point) * scale
 * X/Scale/ZeroPoint 必须已广播到 Y 的元素布局，ZeroPoint 不接受 NULL。
 */
void dequantize_linear_forward(const Tensor* X, const Tensor* Scale, const Tensor* ZeroPoint, Tensor* Y);

/**
 * Conv2D 前向传播
 * 公式: Y = Sum(X * W) + B
 * X/W/Y 采用 NCHW 布局；B 可为 NULL，params 及其数组在调用期间只读借用。
 */
void conv2d_forward(const Tensor* X, const Tensor* W, const Tensor* B, Tensor* Y, ConvParams* params);

/** ConvTranspose 采用 NCHW；B 可为 NULL，输出形状由调用方预先解析。 */
void conv_transpose2d_forward(const Tensor* X, const Tensor* W, const Tensor* B, Tensor* Y, ConvParams* params);

// Col2Im
// 将列块张量按 image_shape/block_shape 累加还原为图像张量。
void col2im_forward(const Tensor* input, const Tensor* image_shape, const Tensor* block_shape,
                    Tensor* output, ConvParams* params);

// DeformConv
// 按 offset 和可选 mask 对输入做双线性采样后执行 2D deformable convolution；B/mask 可为 NULL。
void deform_conv2d_forward(const Tensor* X, const Tensor* W, const Tensor* offset,
                           const Tensor* B, const Tensor* mask, Tensor* Y,
                           ConvParams* params, int offset_group);

// XZeroPoint/WZeroPoint 可为 NULL，分别表示对应输入使用数值零作为零点。
void conv_integer_forward(const Tensor* X, const Tensor* W,
                          const Tensor* XZeroPoint, const Tensor* WZeroPoint,
                          Tensor* Y, ConvParams* params);

// XZeroPoint/WZeroPoint/Bias 可为 NULL；YScale/YZeroPoint 为必需张量。
void qlinear_conv_forward(const Tensor* X, const Tensor* XScale, const Tensor* XZeroPoint,
                          const Tensor* W, const Tensor* WScale, const Tensor* WZeroPoint,
                          const Tensor* YScale, const Tensor* YZeroPoint,
                          const Tensor* Bias, Tensor* Y, ConvParams* params);

// Attention
// Q/K/V/Y 均为 [batch, heads, sequence, head_size]；attn_mask 可为 NULL，并按最多 4D 广播。
// scale < 0 表示使用 1/sqrt(head_size)，softcap <= 0 表示禁用软截断。
void attention_forward(const Tensor* Q, const Tensor* K, const Tensor* V,
                       const Tensor* attn_mask, Tensor* Y,
                       int q_num_heads, int kv_num_heads,
                       float scale, int is_causal, float softcap);
      
/**
 * MaxPool 前向传播
 */
void max_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params);

void max_unpool_forward(const Tensor* X, const Tensor* Indices, Tensor* Y, PoolParams* params);

/**
 * MaxRoiPool for X [N,C,H,W] and rois [num_rois,5].
 */
void max_roi_pool_forward(const Tensor* X, const Tensor* rois, Tensor* Y,
                          int pooled_h, int pooled_w, float spatial_scale);

/**
 * RoiAlign for X [N,C,H,W], rois [num_rois,4], batch_indices [num_rois].
 * mode: 0=avg, 1=max.
 * coordinate_transformation_mode: 0=half_pixel, 1=output_half_pixel.
 */
void roi_align_forward(const Tensor* X, const Tensor* rois, const Tensor* batch_indices, Tensor* Y,
                       int output_height, int output_width, int sampling_ratio,
                       float spatial_scale, int mode, int coordinate_transformation_mode);

/**
 * Gemm (General Matrix Multiply) 前向传播
 * 公式: Y = alpha * A' * B' + beta * C
 * transA/transB: 0=不转置, 1=转置
 * A/B 必须为 2D；C 可为 NULL，并支持标量、行/列向量及末两维广播。
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
 * @param inputs 长度为 num_inputs 的借用指针数组；元素均不可为 NULL。
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
 * min/max 为标量 Tensor；各自可为 NULL，分别表示不设置下界或上界。
 */
void clip_forward(const Tensor* input, Tensor* output, const Tensor* min, const Tensor* max);

/**
 * Cast 类型转换
 * 本质上就是从 Input 读取 (自动转double) 再写入 Output (自动转目标类型)
 */
void cast_forward(const Tensor* input, Tensor* output);

/**
 * BitCast 位重解释
 * 保持底层字节不变，仅按等宽目标 dtype 重新解释输出。
 */
void bitcast_forward(const Tensor* input, Tensor* output);

/**
 * Sum element-wise variadic addition.
 * Python layer prepares all inputs with broadcasted output shape.
 */
void sum_forward(const Tensor** inputs, int num_inputs, Tensor* output);

/**
 * PRelu activation.
 * Python layer prepares X and slope with broadcasted output shape.
 */
void prelu_forward(const Tensor* input, const Tensor* slope, Tensor* output);

/**
 * Batched determinant for tensors shaped [..., M, M].
 */
void det_forward(const Tensor* input, Tensor* output);

/**
 * Flat Unique over all input elements.
 * values/indices/counts 至少预留 input->size 个元素，inverse 至少预留 input->size；
 * 返回唯一值数量，只有各输出的 [0, return_value) 前缀有效（inverse 全长有效）。
 */
int unique_forward(const Tensor* input, Tensor* values, Tensor* indices, Tensor* inverse, Tensor* counts, int sorted);

/**
 * MelWeightMatrix generation.
 */
void mel_weight_matrix_forward(const Tensor* num_mel_bins, const Tensor* dft_length,
                               const Tensor* sample_rate, const Tensor* lower_edge_hertz,
                               const Tensor* upper_edge_hertz, Tensor* output);

/**
 * DFT over a complex-valued tensor represented by a trailing dimension of 1 or 2.
 */
void dft_forward(const Tensor* input, Tensor* output, int axis, int inverse, int onesided, int dft_length);

/**
 * STFT over signal [..., signal_length, 1|2].
 */
void stft_forward(const Tensor* signal, const Tensor* window, Tensor* output,
                  int frame_step, int frame_length, int onesided);

/**
 * Recurrent neural network operators.
 * direction: 0=forward, 1=reverse, 2=bidirectional.
 * layout: 0=[seq,batch,input], 1=[batch,seq,input].
 * activation codes: 0=Tanh, 1=Sigmoid, 2=Relu, 3=Affine, 4=LeakyRelu,
 * 5=ThresholdedRelu, 6=ScaledTanh, 7=HardSigmoid, 8=Elu, 9=Softsign,
 * 10=Softplus. Missing alpha/beta values should be NaN.
 * B、sequence_lens、initial_h、initial_c、P 可按各算子 ONNX 定义传 NULL；Y 为必需，
 * Y_h/Y_c 可为 NULL。activations/alpha/beta 可为 NULL，此时内部采用规范默认激活。
 */
void rnn_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 const int* activations, const float* activation_alpha,
                 const float* activation_beta, int num_activations,
                 float clip, int has_clip);

void gru_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 int linear_before_reset, const int* activations,
                 const float* activation_alpha, const float* activation_beta,
                 int num_activations, float clip, int has_clip);

void lstm_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                  const Tensor* sequence_lens, const Tensor* initial_h,
                  const Tensor* initial_c, const Tensor* P,
                  Tensor* Y, Tensor* Y_h, Tensor* Y_c, int hidden_size,
                  int direction, int layout, int input_forget,
                  const int* activations, const float* activation_alpha,
                  const float* activation_beta, int num_activations,
                  float clip, int has_clip);

/**
 * Multinomial sampling for rank-2 probability tensors.
 */
void multinomial_forward(const Tensor* input, Tensor* output, int sample_size, uint32_t seed);

/**
 * NegativeLogLikelihoodLoss 和 SoftmaxCrossEntropyLoss。
 * reduction: 0=none, 1=mean, 2=sum；weight/weights 可为 NULL，log_prob_output 也可省略。
 */
void negative_log_likelihood_loss_forward(const Tensor* input, const Tensor* target, const Tensor* weight,
                                          Tensor* output, int reduction, int has_ignore_index, int64_t ignore_index);
void softmax_cross_entropy_loss_forward(const Tensor* scores, const Tensor* labels, const Tensor* weights,
                                        Tensor* loss_output, Tensor* log_prob_output,
                                        int reduction, int has_ignore_index, int64_t ignore_index);

/**
 * NonMaxSuppression 将 [batch, class, box] 三元组写入 output。
 * 返回选中行数；调用方应按最坏情况预留容量，并只读取返回行数对应的前缀。
 */
int non_max_suppression_forward(const Tensor* boxes, const Tensor* scores, Tensor* output,
                                int max_output_boxes_per_class, float iou_threshold,
                                float score_threshold, int center_point_box);

/**
 * GridSample for 4D X [N,C,H,W] and grid [N,Hout,Wout,2].
 * mode: 0=bilinear, 1=nearest, 2=bicubic.
 * padding_mode: 0=zeros, 1=border, 2=reflection.
 */
void grid_sample_forward(const Tensor* input, const Tensor* grid, Tensor* output,
                         int mode, int padding_mode, int align_corners);

/**
 * Local Response Normalization over channel dimension 1.
 */
void lrn_forward(const Tensor* input, Tensor* output, int size, float alpha, float beta, float bias);

/**
 * MeanVarianceNormalization: output keeps input shape and normalizes over axes.
 */
void mean_variance_normalization_forward(const Tensor* input, Tensor* output, ReduceParams* params);

/**
 * EyeLike fills output with an identity matrix offset by k.
 */
void eye_like_forward(Tensor* output, int k);

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

/** AZeroPoint/BZeroPoint 可为 NULL，表示使用整数零点 0。 */
void matmul_integer_forward(const Tensor* A, const Tensor* B,
                            const Tensor* AZeroPoint, const Tensor* BZeroPoint,
                            Tensor* Y);

void qlinear_matmul_forward(const Tensor* A, const Tensor* AScale, const Tensor* AZeroPoint,
                            const Tensor* B, const Tensor* BScale, const Tensor* BZeroPoint,
                            const Tensor* YScale, const Tensor* YZeroPoint, Tensor* Y);

/**
 * Gather 算子
 * @param data 输入数据张量
 * @param indices 索引张量 (必须是整数类型)
 * @param output 输出张量
 * @param axis 索引的轴
 */
void gather_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis);

/**
 * Expand 算子 (广播)
 * @param input 输入张量
 * @param output 输出张量 (形状已由 Python 端确定为广播后的形状)
 */
void expand_forward(const Tensor* input, Tensor* output);

/**
 * Shape 算子
 * @param input 输入张量
 * @param output 输出张量 (1D int64，存储 input 的维度)
 */
void shape_forward(const Tensor* input, Tensor* output);

// Constant 复用 flatten_forward

void equal_forward(const Tensor* A, const Tensor* B, Tensor* O);

void greater_forward(const Tensor* A, const Tensor* B, Tensor* O);

void less_forward(const Tensor* A, const Tensor* B, Tensor* O);

void greater_or_equal_forward(const Tensor* A, const Tensor* B, Tensor* O);

void less_or_equal_forward(const Tensor* A, const Tensor* B, Tensor* O);

void not_forward(const Tensor* input, Tensor* output);

void and_forward(const Tensor* A, const Tensor* B, Tensor* O);

void or_forward(const Tensor* A, const Tensor* B, Tensor* O);

void xor_forward(const Tensor* A, const Tensor* B, Tensor* O);

void isnan_forward(const Tensor* input, Tensor* output);

void sin_forward(const Tensor* input, Tensor* output);

void tan_forward(const Tensor* input, Tensor* output);

void atan_forward(const Tensor* input, Tensor* output);

void sign_forward(const Tensor* input, Tensor* output);

void identity_forward(const Tensor* input, Tensor* output);

void mod_forward(const Tensor* A, const Tensor* B, Tensor* O, int fmod_mode);

void where_forward(const Tensor* Cond, const Tensor* X, const Tensor* Y, Tensor* O);

// ConstantOfShape: output 形状已由调用方建立；value 为单元素张量且不可为 NULL。
void constant_of_shape_forward(Tensor* output, const Tensor* value);

// Range
// 生成序列 start, start+delta, ...
void range_forward(const Tensor* start, const Tensor* limit, const Tensor* delta, Tensor* output);

// Tile
// 沿各维度复制
void tile_forward(const Tensor* input, Tensor* output);

// Pad: mode 0=constant, 1=reflect, 2=edge；constant_value 可为 NULL，此时填充值为 0。
void pad_forward(const Tensor* data, Tensor* output, const Tensor* pads, const Tensor* constant_value, int mode);

// CenterCropPad
// 根据 input/output shape 执行居中裁剪或零填充；axes 解析由 Python 层提前体现在 output shape 中。
void center_crop_pad_forward(const Tensor* input, Tensor* output);

// Split 复用 slice

void reduce_mean_forward(const Tensor* input, Tensor* output, ReduceParams* params);

void reduce_sum_forward(const Tensor* input, Tensor* output, ReduceParams* params);

void reduce_max_forward(const Tensor* input, Tensor* output, ReduceParams* params);

void reduce_min_forward(const Tensor* input, Tensor* output, ReduceParams* params);

void reduce_prod_forward(const Tensor* input, Tensor* output, ReduceParams* params);

void argmax_forward(const Tensor* input, Tensor* output, int axis, int select_last_index);

void argmin_forward(const Tensor* input, Tensor* output, int axis, int select_last_index);

// ScatterND 原地修改 data，不生成独立输出；reduction: 0=assignment, 1=add, 2=mul。
void scatter_nd_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int reduction);

// TensorScatter
// 根据 batch 级写入起点更新 KV cache 类张量，mode: 0=linear, 1=circular。
void tensor_scatter_forward(const Tensor* past_cache, const Tensor* update, const Tensor* write_indices, Tensor* output, int axis, int mode);

// GatherND
void gather_nd_forward(const Tensor* data, const Tensor* indices, Tensor* output, int batch_dims);

// GatherElements
void gather_elements_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis);

// NonZero: output 必须是精确预分配的 int64 [input.ndim, num_non_zero]。
void nonzero_forward(const Tensor* input, Tensor* output);

// Resize
// scales 是长度为 input.ndim 的借用数组，输出 shape 已由调用方据此计算。
// mode: 0=nearest, 1=linear
// coord_mode: 0=half_pixel, 1=asymmetric, 2=pytorch_half_pixel, 3=tf_half_pixel_for_nn, 4=align_corners, 5=half_pixel_symmetric
void resize_forward(const Tensor* input, Tensor* output, float* scales, int coord_mode, int mode, int nearest_mode);

// AffineGrid
// 根据 theta 和 size 生成 2D/3D 规范化采样网格
void affine_grid_forward(const Tensor* theta, const Tensor* size, Tensor* output, int align_corners);

// TopK: values/indices 的形状相同且由调用方预分配；indices 必须为 int64，sorted=1 表示排序。
void topk_forward(const Tensor* input, Tensor* values, Tensor* indices, int axis, int largest, int sorted, int K);

// CumSum
// exclusive: 0=False(default), 1=True
// reverse: 0=False(default), 1=True
void cumsum_forward(const Tensor* input, Tensor* output, int axis, int exclusive, int reverse);

// CumProd
// exclusive: 0=False(default), 1=True
// reverse: 0=False(default), 1=True
void cumprod_forward(const Tensor* input, Tensor* output, int axis, int exclusive, int reverse);

// RandomUniformLike
// 生成均匀分布 [low, high)
void random_uniform_like_forward(Tensor* output, float low, float high, float seed);

// Einsum (广义爱因斯坦求和) 使用调用方预编译的循环步长表执行任意维度缩并。
// iter_dims: 循环的总维度数 (即方程中唯一标签的数量)
// loop_limits: 每个循环维度的上限 [dim0, dim1, ...]
// input_strides: 展平后的输入步长表。大小 = num_inputs * iter_dims
// output_strides: 输出步长表。大小 = iter_dims
void einsum_forward(const Tensor** inputs, int num_inputs, Tensor* output, 
                    int iter_dims, int* loop_limits, 
                    int* input_strides, int* output_strides);

// 以下逐元素激活函数均保持元素数量，output 的 shape/dtype 由调用方预先确定。
void elu_forward(const Tensor* input, Tensor* output, float alpha);
void selu_forward(const Tensor* input, Tensor* output, float alpha, float gamma);
void leaky_relu_forward(const Tensor* input, Tensor* output, float alpha);
void thresholded_relu_forward(const Tensor* input, Tensor* output, float alpha);
void hard_sigmoid_forward(const Tensor* input, Tensor* output, float alpha, float beta);
void softplus_forward(const Tensor* input, Tensor* output);
void softsign_forward(const Tensor* input, Tensor* output);
void celu_forward(const Tensor* input, Tensor* output, float alpha);
void hard_swish_forward(const Tensor* input, Tensor* output);
void swish_forward(const Tensor* input, Tensor* output, float alpha);
void shrink_forward(const Tensor* input, Tensor* output, float bias, float lambd);
void acos_forward(const Tensor* input, Tensor* output);
void asin_forward(const Tensor* input, Tensor* output);
void cosh_forward(const Tensor* input, Tensor* output);
void sinh_forward(const Tensor* input, Tensor* output);
void asinh_forward(const Tensor* input, Tensor* output);
void acosh_forward(const Tensor* input, Tensor* output);
void atanh_forward(const Tensor* input, Tensor* output);
 void bitwise_and_forward(const Tensor* A, const Tensor* B, Tensor* O);
void bitwise_or_forward(const Tensor* A, const Tensor* B, Tensor* O);
void bitwise_xor_forward(const Tensor* A, const Tensor* B, Tensor* O);
void bitwise_not_forward(const Tensor* input, Tensor* output);
// 位运算要求输入/输出为兼容整数 dtype；direction: 0=LEFT, 1=RIGHT。
void bit_shift_forward(const Tensor* A, const Tensor* B, Tensor* O, int direction);
void reduce_l1_forward(const Tensor* input, Tensor* output, ReduceParams* params);
void reduce_l2_forward(const Tensor* input, Tensor* output, ReduceParams* params);
void reduce_log_sum_forward(const Tensor* input, Tensor* output, ReduceParams* params);
void reduce_log_sum_exp_forward(const Tensor* input, Tensor* output, ReduceParams* params);
void reduce_sum_square_forward(const Tensor* input, Tensor* output, ReduceParams* params);
void average_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params, int count_include_pad);
void lp_pool_forward(const Tensor* X, Tensor* Y, PoolParams* params, int p);
void global_average_pool_forward(const Tensor* input, Tensor* output);
void global_max_pool_forward(const Tensor* input, Tensor* output);
void global_lp_pool_forward(const Tensor* input, Tensor* output, int p);
void mean_forward(const Tensor** inputs, int num_inputs, Tensor* output);
void size_forward(const Tensor* input, Tensor* output);
void isinf_forward(const Tensor* input, Tensor* output, int detect_pos, int detect_neg);
void one_hot_forward(const Tensor* indices, const Tensor* values, Tensor* output, int axis);
void triangular_forward(const Tensor* input, Tensor* output, int k, int upper);

// BatchNormalization: 推理入口只写 output；训练入口额外写 running_mean/running_var。
void batch_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        const Tensor* mean, const Tensor* var, Tensor* output, float epsilon);
void batch_norm_training_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                 const Tensor* mean, const Tensor* var,
                                 Tensor* output, Tensor* running_mean, Tensor* running_var,
                                 float epsilon, float momentum);

// InstanceNormalization
void instance_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                           Tensor* output, float epsilon);

// LayerNormalization: axis 之前为外层维，axis 到末维参与归一化；scale/B 可为 NULL。
// 多输出入口的 mean_output/inv_std_output 为调用方预分配的统计量缓冲区。
void layer_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int axis, float epsilon);
void layer_norm_multi_output_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                     Tensor* output, Tensor* mean_output, Tensor* inv_std_output,
                                     int axis, float epsilon);

// RMSNormalization
// axis: 从该轴到最后一维计算 root mean square；scale 按 ONNX 单向广播规则映射。
void rms_normalization_forward(const Tensor* input, const Tensor* scale, Tensor* output,
                               int axis, float epsilon, int stash_type);

// RotaryEmbedding
// 按 ONNX RoPE 语义旋转 3D/4D 输入的每个 head 前缀维度。
void rotary_embedding_forward(const Tensor* input, const Tensor* cos_cache, const Tensor* sin_cache,
                              const Tensor* position_ids, Tensor* output,
                              int num_heads, int rotary_embedding_dim, int interleaved);

void round_forward(const Tensor* input, Tensor* output);
void erf_forward(const Tensor* input, Tensor* output);


// 窗函数
// periodic: 1=True (period N), 0=False (period N-1, symmetric)
void hann_window_forward(const Tensor* size_tensor, Tensor* output, int periodic);
void hamming_window_forward(const Tensor* size_tensor, Tensor* output, int periodic);
void blackman_window_forward(const Tensor* size_tensor, Tensor* output, int periodic);

// 随机生成器只写预分配 output；seed 以数值形式传入，由各实现转换为内部状态。
void random_normal_forward(Tensor* output, float mean, float scale, float seed);
void bernoulli_forward(const Tensor* input, Tensor* output, float seed); 

// Dropout: ratio 为置零概率；training_mode=0 直接复制，=1 时随机置零并除以 (1-ratio)。
// 该 C 入口只产生数据输出，不产生 ONNX 的可选 mask 输出。
void dropout_forward(const Tensor* input, Tensor* output, float ratio, int training_mode);

// Gelu 精确公式：0.5 * x * (1 + erf(x / sqrt(2)))
void gelu_forward(const Tensor* input, Tensor* output);

// Gelu 带近似模式入口：0=none，1=tanh。
void gelu_forward_mode(const Tensor* input, Tensor* output, int approximate_mode);

// Mish Activation: x * tanh(ln(1 + e^x))
void mish_forward(const Tensor* input, Tensor* output);

// Hardmax: One-hot based on argmax along axis
void hardmax_forward(const Tensor* input, Tensor* output, int axis);

// LogSoftmax: log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
void log_softmax_forward(const Tensor* input, Tensor* output, int axis);

// LpNormalization: x / ||x||_p
// p: norm degree (usually 1 or 2)
void lp_normalization_forward(const Tensor* input, Tensor* output, int axis, int p);

// DepthToSpace
// mode: 0=DCR (Depth-Column-Row), 1=CRD (Column-Row-Depth)
void depth_to_space_forward(const Tensor* input, Tensor* output, int blocksize, int mode);

// SpaceToDepth
void space_to_depth_forward(const Tensor* input, Tensor* output, int blocksize);

// ReverseSequence
void reverse_sequence_forward(const Tensor* input, const Tensor* sequence_lens, Tensor* output, int time_axis, int batch_axis);

// Compress: output 必须按 condition 筛选后的精确形状预分配。
void compress_forward(const Tensor* input, const Tensor* condition, Tensor* output, int axis);

// ScatterElements 原地修改 data；reduction 编码与 ScatterND 相同。
void scatter_elements_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int axis, int reduction);

// GroupNormalization
void group_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int num_groups, float epsilon);

// Binarizer
// val > threshold ? 1 : 0
void binarizer_forward(const Tensor* input, Tensor* output, float threshold);

// DynamicQuantizeLinear 同时写 y、标量 y_scale 和标量 y_zp，三者均由调用方预分配。
void dynamic_quantize_linear_forward(const Tensor* x, Tensor* y, Tensor* y_scale, Tensor* y_zp);

#endif
