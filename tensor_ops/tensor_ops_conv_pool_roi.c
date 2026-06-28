/**
  ******************************************************************************
  * @file        tensor_ops_conv_pool_roi.c
  * @author      Egor Izmaylov
  * @brief       实现卷积、Col2Im、DeformConv 和量化卷积类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `conv2d` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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
// 实现 `conv transpose2d` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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

// 将线性下标展开成指定形状的坐标，用于 Col2Im 的 kernel/block 坐标映射。
static void col2im_unravel_index(size_t index, const int* shape, int rank, int* coords) {
    for (int axis = rank - 1; axis >= 0; --axis) {
        coords[axis] = (int)(index % (size_t)shape[axis]);
        index /= (size_t)shape[axis];
    }
}


// 实现 `col2im` 算子的 C 后端入口，按 N-D fold 语义把列块累加回图像。
void col2im_forward(const Tensor* input, const Tensor* image_shape, const Tensor* block_shape,
                    Tensor* output, ConvParams* params) {
    if (!input || !image_shape || !block_shape || !output || !params) return;
    if (!input->data || !image_shape->data || !block_shape->data || !output->data) return;
    if (input->ndim != 3 || output->ndim < 4 || output->ndim > MAX_NDIM) return;

    int spatial_rank = output->ndim - 2;
    if (image_shape->size != (size_t)spatial_rank || block_shape->size != (size_t)spatial_rank) return;

    int image_dims[MAX_NDIM] = {0};
    int block_dims[MAX_NDIM] = {0};
    int n_blocks[MAX_NDIM] = {0};

    size_t kernel_size = 1;
    size_t block_count = 1;
    for (int axis = 0; axis < spatial_rank; ++axis) {
        image_dims[axis] = (int)get_value_as_int64(image_shape, (size_t)axis);
        block_dims[axis] = (int)get_value_as_int64(block_shape, (size_t)axis);
        if (image_dims[axis] <= 0 || block_dims[axis] <= 0) return;
        int pad_begin = params->pads ? params->pads[axis] : 0;
        int pad_end = params->pads ? params->pads[axis + spatial_rank] : 0;
        int stride = params->strides ? params->strides[axis] : 1;
        int dilation = params->dilations ? params->dilations[axis] : 1;
        if (stride <= 0 || dilation <= 0) return;
        n_blocks[axis] = (image_dims[axis] + pad_begin + pad_end - dilation * (block_dims[axis] - 1) - 1) / stride + 1;
        if (n_blocks[axis] <= 0) return;
        kernel_size *= (size_t)block_dims[axis];
        block_count *= (size_t)n_blocks[axis];
        if (output->shape[axis + 2] != image_dims[axis]) return;
    }

    int batch = input->shape[0];
    int planes = input->shape[1];
    int columns = input->shape[2];
    if ((size_t)columns != block_count || kernel_size == 0 || planes % (int)kernel_size != 0) return;
    int channels = planes / (int)kernel_size;
    if (output->shape[0] != batch || output->shape[1] != channels) return;

    double* accum = (double*)calloc(output->size == 0 ? 1 : output->size, sizeof(double));
    if (!accum) return;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            int local_kernel_coords[MAX_NDIM] = {0};
            int local_block_coords[MAX_NDIM] = {0};
            for (size_t k = 0; k < kernel_size; ++k) {
                col2im_unravel_index(k, block_dims, spatial_rank, local_kernel_coords);
                for (size_t col = 0; col < block_count; ++col) {
                    col2im_unravel_index(col, n_blocks, spatial_rank, local_block_coords);
                    size_t out_spatial_index = 0;
                    int inside = 1;
                    for (int axis = 0; axis < spatial_rank; ++axis) {
                        int pad_begin = params->pads ? params->pads[axis] : 0;
                        int stride = params->strides ? params->strides[axis] : 1;
                        int dilation = params->dilations ? params->dilations[axis] : 1;
                        int image_coord = local_block_coords[axis] * stride - pad_begin + local_kernel_coords[axis] * dilation;
                        if (image_coord < 0 || image_coord >= image_dims[axis]) {
                            inside = 0;
                            break;
                        }
                        out_spatial_index = out_spatial_index * (size_t)image_dims[axis] + (size_t)image_coord;
                    }
                    if (!inside) continue;

                    size_t input_index = ((size_t)n * (size_t)planes + (size_t)c * kernel_size + k) * block_count + col;
                    size_t output_index = ((size_t)n * (size_t)channels + (size_t)c);
                    for (int axis = 0; axis < spatial_rank; ++axis) output_index *= (size_t)image_dims[axis];
                    output_index += out_spatial_index;
                    double value = get_value_as_double(input, input_index);
                    #pragma omp atomic
                    accum[output_index] += value;
                }
            }
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; ++i) {
        set_tensor_value_from_float(output, i, accum[i]);
    }
    free(accum);
}
