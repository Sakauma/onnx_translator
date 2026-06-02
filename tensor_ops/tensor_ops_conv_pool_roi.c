/*
 * 文件功能：实现卷积、池化、反池化和 ROI 类 C 后端算子。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
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


// 实现 `conv integer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void conv_integer_forward(const Tensor* X, const Tensor* W,
                          const Tensor* XZeroPoint, const Tensor* WZeroPoint,
                          Tensor* Y, ConvParams* params) {
    if (!X || !W || !Y || !params || X->ndim != 4 || W->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c  = X->shape[1];
    int out_c = W->shape[0];
    int k_h   = W->shape[2];
    int k_w   = W->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    if (group <= 0) return;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;

    _Pragma("omp parallel for collapse(2)")
    for (int n = 0; n < batch; n++) {
        for (int m = 0; m < out_c; m++) {
            int g = m / out_c_per_group;
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int64_t sum = 0;
                    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
                        int ic = g * in_c_per_group + ic_g;
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                int h_in = oh * stride_h + kh * dilation_h - pad_top;
                                int w_in = ow * stride_w + kw * dilation_w - pad_left;
                                if (h_in < 0 || h_in >= X->shape[2] || w_in < 0 || w_in >= X->shape[3]) {
                                    continue;
                                }

                                size_t x_idx = ((size_t)n * in_c * X->shape[2] * X->shape[3]) +
                                               ((size_t)ic * X->shape[2] * X->shape[3]) +
                                               ((size_t)h_in * X->shape[3]) + w_in;
                                size_t w_idx = ((size_t)m * in_c_per_group * k_h * k_w) +
                                               ((size_t)ic_g * k_h * k_w) +
                                               ((size_t)kh * k_w) + kw;

                                int64_t x_val = get_value_as_int64(X, x_idx);
                                int64_t w_val = get_value_as_int64(W, w_idx);
                                int64_t x_zp = (XZeroPoint && XZeroPoint->data) ? get_value_as_int64(XZeroPoint, x_idx) : 0;
                                int64_t w_zp = (WZeroPoint && WZeroPoint->data) ? get_value_as_int64(WZeroPoint, w_idx) : 0;
                                sum += (x_val - x_zp) * (w_val - w_zp);
                            }
                        }
                    }

                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)m * out_h * out_w) +
                                   ((size_t)oh * out_w) + ow;
                    if (Y->dtype == DTYPE_INT32) {
                        ((int32_t*)Y->data)[y_idx] = (int32_t)sum;
                    } else {
                        set_tensor_value_from_int(Y, y_idx, sum);
                    }
                }
            }
        }
    }
}


// 实现 `qlinear conv` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void qlinear_conv_forward(const Tensor* X, const Tensor* XScale, const Tensor* XZeroPoint,
                          const Tensor* W, const Tensor* WScale, const Tensor* WZeroPoint,
                          const Tensor* YScale, const Tensor* YZeroPoint,
                          const Tensor* Bias, Tensor* Y, ConvParams* params) {
    if (!X || !XScale || !W || !WScale || !YScale || !YZeroPoint || !Y || !params) return;
    if (X->ndim != 4 || W->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c  = X->shape[1];
    int out_c = W->shape[0];
    int k_h   = W->shape[2];
    int k_w   = W->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];

    int pad_top = params->pads[0];
    int pad_left = params->pads[1];
    int stride_h = params->strides[0];
    int stride_w = params->strides[1];
    int dilation_h = params->dilations[0];
    int dilation_w = params->dilations[1];
    int group = params->group;
    if (group <= 0 || in_c % group != 0 || out_c % group != 0) return;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;

    _Pragma("omp parallel for collapse(2)")
    for (int n = 0; n < batch; n++) {
        for (int m = 0; m < out_c; m++) {
            int g = m / out_c_per_group;
            size_t w_scale_idx = (size_t)m * in_c_per_group * k_h * k_w;
            double w_scale = get_value_as_double(WScale, w_scale_idx);
            int64_t bias = (Bias && Bias->data) ? get_value_as_int64(Bias, m) : 0;

            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int64_t acc = 0;
                    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
                        int ic = g * in_c_per_group + ic_g;
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                int h_in = oh * stride_h + kh * dilation_h - pad_top;
                                int w_in = ow * stride_w + kw * dilation_w - pad_left;
                                if (h_in < 0 || h_in >= X->shape[2] || w_in < 0 || w_in >= X->shape[3]) {
                                    continue;
                                }

                                size_t x_idx = ((size_t)n * in_c * X->shape[2] * X->shape[3]) +
                                               ((size_t)ic * X->shape[2] * X->shape[3]) +
                                               ((size_t)h_in * X->shape[3]) + w_in;
                                size_t w_idx = ((size_t)m * in_c_per_group * k_h * k_w) +
                                               ((size_t)ic_g * k_h * k_w) +
                                               ((size_t)kh * k_w) + kw;

                                int64_t x_val = get_value_as_int64(X, x_idx);
                                int64_t w_val = get_value_as_int64(W, w_idx);
                                int64_t x_zp = (XZeroPoint && XZeroPoint->data) ? get_value_as_int64(XZeroPoint, x_idx) : 0;
                                int64_t w_zp = (WZeroPoint && WZeroPoint->data) ? get_value_as_int64(WZeroPoint, w_idx) : 0;
                                acc += (x_val - x_zp) * (w_val - w_zp);
                            }
                        }
                    }

                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)m * out_h * out_w) +
                                   ((size_t)oh * out_w) + ow;
                    double x_scale = get_value_as_double(XScale, 0);
                    double y_scale = get_value_as_double(YScale, y_idx);
                    double y_zp = get_value_as_double(YZeroPoint, y_idx);
                    double q = ((double)(acc + bias) * x_scale * w_scale) / y_scale + y_zp;
                    set_tensor_value_from_float(Y, y_idx, q);
                }
            }
        }
    }
}


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


// 实现 `max roi pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void max_roi_pool_forward(const Tensor* X, const Tensor* rois, Tensor* Y,
                          int pooled_h, int pooled_w, float spatial_scale) {
    if (!X || !rois || !Y || !X->data || !rois->data || !Y->data) return;
    if (X->ndim != 4 || rois->ndim != 2 || Y->ndim != 4 || rois->shape[1] != 5) return;
    if (pooled_h <= 0 || pooled_w <= 0) return;

    int num_rois = rois->shape[0];
    int batches = X->shape[0];
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (Y->shape[0] != num_rois || Y->shape[1] != channels ||
        Y->shape[2] != pooled_h || Y->shape[3] != pooled_w) return;

    _Pragma("omp parallel for collapse(4)")
    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < pooled_h; ph++) {
                for (int pw = 0; pw < pooled_w; pw++) {
                    size_t roi_base = (size_t)roi_idx * 5;
                    int batch = (int)get_value_as_int64(rois, roi_base);
                    if (batch < 0 || batch >= batches) continue;

                    int x1 = (int)nearbyint(get_value_as_double(rois, roi_base + 1) * (double)spatial_scale);
                    int y1 = (int)nearbyint(get_value_as_double(rois, roi_base + 2) * (double)spatial_scale);
                    int x2 = (int)nearbyint(get_value_as_double(rois, roi_base + 3) * (double)spatial_scale);
                    int y2 = (int)nearbyint(get_value_as_double(rois, roi_base + 4) * (double)spatial_scale);

                    int roi_w = x2 - x1 + 1;
                    int roi_h = y2 - y1 + 1;
                    if (roi_w < 1) roi_w = 1;
                    if (roi_h < 1) roi_h = 1;
                    double bin_h = (double)roi_h / (double)pooled_h;
                    double bin_w = (double)roi_w / (double)pooled_w;

                    int hstart = (int)floor((double)ph * bin_h) + y1;
                    int hend = (int)ceil((double)(ph + 1) * bin_h) + y1;
                    int wstart = (int)floor((double)pw * bin_w) + x1;
                    int wend = (int)ceil((double)(pw + 1) * bin_w) + x1;
                    if (hstart < 0) hstart = 0;
                    if (hend < 0) hend = 0;
                    if (wstart < 0) wstart = 0;
                    if (wend < 0) wend = 0;
                    if (hstart > height) hstart = height;
                    if (hend > height) hend = height;
                    if (wstart > width) wstart = width;
                    if (wend > width) wend = width;

                    double max_val = 0.0;
                    if (hend > hstart && wend > wstart) {
                        max_val = -DBL_MAX;
                        for (int h = hstart; h < hend; h++) {
                            for (int w = wstart; w < wend; w++) {
                                size_t x_idx = ((size_t)batch * channels * height * width)
                                             + ((size_t)c * height * width)
                                             + ((size_t)h * width)
                                             + (size_t)w;
                                double value = get_value_as_double(X, x_idx);
                                if (value > max_val) max_val = value;
                            }
                        }
                    }

                    size_t y_idx = ((size_t)roi_idx * channels * pooled_h * pooled_w)
                                 + ((size_t)c * pooled_h * pooled_w)
                                 + ((size_t)ph * pooled_w)
                                 + (size_t)pw;
                    set_tensor_value_from_float(Y, y_idx, max_val);
                }
            }
        }
    }
}


// 实现 `roi align` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void roi_align_forward(const Tensor* X, const Tensor* rois, const Tensor* batch_indices, Tensor* Y,
                       int output_height, int output_width, int sampling_ratio,
                       float spatial_scale, int mode, int coordinate_transformation_mode) {
    if (!X || !rois || !batch_indices || !Y || !X->data || !rois->data || !batch_indices->data || !Y->data) return;
    if (X->ndim != 4 || rois->ndim != 2 || batch_indices->ndim != 1 || Y->ndim != 4 || rois->shape[1] != 4) return;
    if (output_height <= 0 || output_width <= 0) return;

    int num_rois = rois->shape[0];
    int batches = X->shape[0];
    int channels = X->shape[1];
    if (batch_indices->shape[0] != num_rois ||
        Y->shape[0] != num_rois || Y->shape[1] != channels ||
        Y->shape[2] != output_height || Y->shape[3] != output_width) return;

    int half_pixel = (coordinate_transformation_mode == 0);
    double offset = half_pixel ? 0.5 : 0.0;

    _Pragma("omp parallel for collapse(4)")
    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < output_height; ph++) {
                for (int pw = 0; pw < output_width; pw++) {
                    int batch = (int)get_value_as_int64(batch_indices, (size_t)roi_idx);
                    if (batch < 0 || batch >= batches) continue;

                    size_t roi_base = (size_t)roi_idx * 4;
                    double roi_start_w = get_value_as_double(rois, roi_base) * (double)spatial_scale - offset;
                    double roi_start_h = get_value_as_double(rois, roi_base + 1) * (double)spatial_scale - offset;
                    double roi_end_w = get_value_as_double(rois, roi_base + 2) * (double)spatial_scale - offset;
                    double roi_end_h = get_value_as_double(rois, roi_base + 3) * (double)spatial_scale - offset;
                    double roi_w = roi_end_w - roi_start_w;
                    double roi_h = roi_end_h - roi_start_h;
                    if (!half_pixel) {
                        if (roi_w < 1.0) roi_w = 1.0;
                        if (roi_h < 1.0) roi_h = 1.0;
                    }
                    double bin_h = roi_h / (double)output_height;
                    double bin_w = roi_w / (double)output_width;
                    int grid_h = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_h / (double)output_height);
                    int grid_w = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_w / (double)output_width);
                    if (grid_h < 1) grid_h = 1;
                    if (grid_w < 1) grid_w = 1;
                    int count = grid_h * grid_w;

                    double output_value = (mode == 1) ? -DBL_MAX : 0.0;
                    for (int iy = 0; iy < grid_h; iy++) {
                        double yy = roi_start_h + (double)ph * bin_h + ((double)iy + 0.5) * bin_h / (double)grid_h;
                        for (int ix = 0; ix < grid_w; ix++) {
                            double xx = roi_start_w + (double)pw * bin_w + ((double)ix + 0.5) * bin_w / (double)grid_w;
                            if (mode == 1) {
                                double term = roi_align_max_weighted_term(X, batch, c, yy, xx);
                                if (term > output_value) output_value = term;
                            } else {
                                output_value += roi_align_bilinear_sample(X, batch, c, yy, xx);
                            }
                        }
                    }
                    if (mode != 1) {
                        output_value /= (double)count;
                    }

                    size_t y_idx = ((size_t)roi_idx * channels * output_height * output_width)
                                 + ((size_t)c * output_height * output_width)
                                 + ((size_t)ph * output_width)
                                 + (size_t)pw;
                    set_tensor_value_from_float(Y, y_idx, output_value);
                }
            }
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


// GlobalAveragePool
// 假设输入是 NCHW (或至少后两维是空间维度)，如果不符合则不执行
// 实现 `global average pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_average_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            sum += get_value_as_double(input, offset + i);
        }
        set_tensor_value_from_float(output, n, sum / spatial_size);
    }
}


// GlobalMaxPool
// 实现 `global max pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_max_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double max_val = -DBL_MAX;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            if (val > max_val) max_val = val;
        }
        set_tensor_value_from_float(output, n, max_val);
    }
}


// GlobalLpPool
// 实现 `global lp pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_lp_pool_forward(const Tensor* input, Tensor* output, int p) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;
    
    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];
    
    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum_pow = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            sum_pow += pow(fabs(val), p);
        }
        
        // p=1 时就是 Sum(|x|)，p=2 时是 L2 Norm，p=inf 时是 Max
        double res = pow(sum_pow, 1.0 / p);
        set_tensor_value_from_float(output, n, res);
    }
}

