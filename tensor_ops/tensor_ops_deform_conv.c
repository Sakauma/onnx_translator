/**
  ******************************************************************************
  * @file        tensor_ops_deform_conv.c
  * @author      Egor Izmaylov
  * @brief       实现 DeformConv 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从卷积 shard 拆分 DeformConv 热点实现。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 对实际图像坐标执行 zeros padding 的双线性采样，匹配 GridSample align_corners=1 的 DeformConv 用法。
static double deform_conv_bilinear_sample(const Tensor* X, int n, int c, double y, double x) {
    int height = X->shape[2];
    int width = X->shape[3];
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    double wy1 = y - (double)y0;
    double wx1 = x - (double)x0;
    double wy0 = 1.0 - wy1;
    double wx0 = 1.0 - wx1;
    double value = 0.0;

    if (y0 >= 0 && y0 < height && x0 >= 0 && x0 < width) {
        size_t idx = ((size_t)n * X->shape[1] * height * width) +
                     ((size_t)c * height * width) +
                     ((size_t)y0 * width) + (size_t)x0;
        value += wy0 * wx0 * get_value_as_double(X, idx);
    }
    if (y0 >= 0 && y0 < height && x1 >= 0 && x1 < width) {
        size_t idx = ((size_t)n * X->shape[1] * height * width) +
                     ((size_t)c * height * width) +
                     ((size_t)y0 * width) + (size_t)x1;
        value += wy0 * wx1 * get_value_as_double(X, idx);
    }
    if (y1 >= 0 && y1 < height && x0 >= 0 && x0 < width) {
        size_t idx = ((size_t)n * X->shape[1] * height * width) +
                     ((size_t)c * height * width) +
                     ((size_t)y1 * width) + (size_t)x0;
        value += wy1 * wx0 * get_value_as_double(X, idx);
    }
    if (y1 >= 0 && y1 < height && x1 >= 0 && x1 < width) {
        size_t idx = ((size_t)n * X->shape[1] * height * width) +
                     ((size_t)c * height * width) +
                     ((size_t)y1 * width) + (size_t)x1;
        value += wy1 * wx1 * get_value_as_double(X, idx);
    }
    return value;
}


// 实现 `DeformConv` 的 2D C 后端入口，覆盖 group、offset_group、bias 和 mask 主语义。
void deform_conv2d_forward(const Tensor* X, const Tensor* W, const Tensor* offset,
                           const Tensor* B, const Tensor* mask, Tensor* Y,
                           ConvParams* params, int offset_group) {
    if (!X || !W || !offset || !Y || !params) return;
    if (!X->data || !W->data || !offset->data || !Y->data) return;
    if (X->ndim != 4 || W->ndim != 4 || offset->ndim != 4 || Y->ndim != 4) return;

    int batch = X->shape[0];
    int in_c = X->shape[1];
    int in_h = X->shape[2];
    int in_w = X->shape[3];
    int out_c = W->shape[0];
    int weight_c = W->shape[1];
    int k_h = W->shape[2];
    int k_w = W->shape[3];
    int out_h = Y->shape[2];
    int out_w = Y->shape[3];
    int group = params->group;
    if (group <= 0 || offset_group <= 0) return;
    if (batch != Y->shape[0] || out_c != Y->shape[1]) return;
    if (in_c != weight_c * group || out_c % group != 0 || in_c % offset_group != 0) return;
    if (offset->shape[0] != batch || offset->shape[2] != out_h || offset->shape[3] != out_w) return;
    if (offset->shape[1] != offset_group * k_h * k_w * 2) return;
    if (mask && mask->data) {
        if (mask->ndim != 4 || mask->shape[0] != batch || mask->shape[1] != offset_group * k_h * k_w ||
            mask->shape[2] != out_h || mask->shape[3] != out_w) return;
    }
    if (B && B->data && B->size != (size_t)out_c) return;

    int pad_h = params->pads ? params->pads[0] : 0;
    int pad_w = params->pads ? params->pads[1] : 0;
    int stride_h = params->strides ? params->strides[0] : 1;
    int stride_w = params->strides ? params->strides[1] : 1;
    int dilation_h = params->dilations ? params->dilations[0] : 1;
    int dilation_w = params->dilations ? params->dilations[1] : 1;
    if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0) return;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;
    int in_c_per_offset_group = in_c / offset_group;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_c; ++oc) {
            int conv_group = oc / out_c_per_group;
            int ic_begin = conv_group * in_c_per_group;
            int ic_end = ic_begin + in_c_per_group;
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double sum = (B && B->data) ? get_value_as_double(B, (size_t)oc) : 0.0;
                    for (int ic = ic_begin; ic < ic_end; ++ic) {
                        int oc_local = ic - ic_begin;
                        int offset_group_idx = ic / in_c_per_offset_group;
                        for (int kh = 0; kh < k_h; ++kh) {
                            for (int kw = 0; kw < k_w; ++kw) {
                                int kernel_linear = kh * k_w + kw;
                                int offset_base_c = ((offset_group_idx * k_h + kh) * k_w + kw) * 2;
                                size_t offset_h_idx = ((size_t)n * offset->shape[1] * out_h * out_w) +
                                                      ((size_t)offset_base_c * out_h * out_w) +
                                                      ((size_t)oh * out_w) + (size_t)ow;
                                size_t offset_w_idx = offset_h_idx + (size_t)out_h * out_w;
                                double sample_y = -pad_h + oh * stride_h + kh * dilation_h + get_value_as_double(offset, offset_h_idx);
                                double sample_x = -pad_w + ow * stride_w + kw * dilation_w + get_value_as_double(offset, offset_w_idx);
                                double sampled = deform_conv_bilinear_sample(X, n, ic, sample_y, sample_x);
                                double mask_value = 1.0;
                                if (mask && mask->data) {
                                    int mask_c = offset_group_idx * k_h * k_w + kernel_linear;
                                    size_t mask_idx = ((size_t)n * mask->shape[1] * out_h * out_w) +
                                                      ((size_t)mask_c * out_h * out_w) +
                                                      ((size_t)oh * out_w) + (size_t)ow;
                                    mask_value = get_value_as_double(mask, mask_idx);
                                }
                                size_t w_idx = ((size_t)oc * weight_c * k_h * k_w) +
                                               ((size_t)oc_local * k_h * k_w) +
                                               ((size_t)kh * k_w) + (size_t)kw;
                                sum += sampled * get_value_as_double(W, w_idx) * mask_value;
                            }
                        }
                    }
                    size_t y_idx = ((size_t)n * out_c * out_h * out_w) +
                                   ((size_t)oc * out_h * out_w) +
                                   ((size_t)oh * out_w) + (size_t)ow;
                    set_tensor_value_from_float(Y, y_idx, sum);
                }
            }
        }
    }
    (void)in_h;
    (void)in_w;
}
