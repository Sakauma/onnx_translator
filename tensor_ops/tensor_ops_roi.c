/**
  ******************************************************************************
  * @file        tensor_ops_roi.c
  * @author      Egor Izmaylov
  * @brief       实现 ROI Pool 和 ROI Align 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从局部池化 shard 拆分 ROI 实现。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "internal/roi.h"


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
