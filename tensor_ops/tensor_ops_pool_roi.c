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
