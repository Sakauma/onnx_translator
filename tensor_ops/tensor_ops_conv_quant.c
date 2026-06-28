/**
  ******************************************************************************
  * @file        tensor_ops_conv_quant.c
  * @author      Egor Izmaylov
  * @brief       实现量化卷积类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


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
