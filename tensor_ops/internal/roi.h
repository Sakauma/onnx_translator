/**
  ******************************************************************************
  * @file        roi.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_ROI_H
#define TENSOR_OPS_INTERNAL_ROI_H

#include "../tensor_ops_internal.h"

// 实现 `roi_align_bilinear_sample` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double roi_align_bilinear_sample(const Tensor* X, int batch, int channel, double y, double x) {
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (y < -1.0 || y > (double)height || x < -1.0 || x > (double)width) {
        return 0.0;
    }
    if (y < 0.0) y = 0.0;
    if (x < 0.0) x = 0.0;
    int y0 = (int)y;
    int x0 = (int)x;
    int y1;
    int x1;
    if (y0 >= height - 1) {
        y1 = y0 = height - 1;
        y = (double)y0;
    } else {
        y1 = y0 + 1;
    }
    if (x0 >= width - 1) {
        x1 = x0 = width - 1;
        x = (double)x0;
    } else {
        x1 = x0 + 1;
    }
    double ly = y - (double)y0;
    double lx = x - (double)x0;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    double total = 0.0;
    int ys[2] = {y0, y1};
    int xs[2] = {x0, x1};
    double wy[2] = {hy, ly};
    double wx[2] = {hx, lx};
    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            size_t idx = ((size_t)batch * channels * height * width)
                       + ((size_t)channel * height * width)
                       + ((size_t)ys[iy] * width)
                       + (size_t)xs[ix];
            total += get_value_as_double(X, idx) * wy[iy] * wx[ix];
        }
    }
    return total;
}

// 实现 `roi_align_max_weighted_term` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double roi_align_max_weighted_term(const Tensor* X, int batch, int channel, double y, double x) {
    int channels = X->shape[1];
    int height = X->shape[2];
    int width = X->shape[3];
    if (y < -1.0 || y > (double)height || x < -1.0 || x > (double)width) {
        return 0.0;
    }
    if (y < 0.0) y = 0.0;
    if (x < 0.0) x = 0.0;
    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;
    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (double)y_low;
    } else {
        y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (double)x_low;
    } else {
        x_high = x_low + 1;
    }
    double ly = y - (double)y_low;
    double lx = x - (double)x_low;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    int ys[2] = {y_low, y_high};
    int xs[2] = {x_low, x_high};
    double wy[2] = {hy, ly};
    double wx[2] = {hx, lx};
    double max_term = -DBL_MAX;
    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            size_t idx = ((size_t)batch * channels * height * width)
                       + ((size_t)channel * height * width)
                       + ((size_t)ys[iy] * width)
                       + (size_t)xs[ix];
            double term = get_value_as_double(X, idx) * wy[iy] * wx[ix];
            if (term > max_term) max_term = term;
        }
    }
    return max_term;
}

#endif /* TENSOR_OPS_INTERNAL_ROI_H */
