/**
  ******************************************************************************
  * @file        tensor_ops_pool_roi.c
  * @author      Egor Izmaylov
  * @brief       实现局部池化、反池化和 ROI 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 实现 `max pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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


// 实现 `max unpool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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




// AveragePool
// 实现 `average pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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
// 实现 `lp pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
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
