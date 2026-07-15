/**
  ******************************************************************************
  * @file        detection_sampling.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_DETECTION_SAMPLING_H
#define TENSOR_OPS_INTERNAL_DETECTION_SAMPLING_H

#include "../tensor_ops_internal.h"

// 实现 `nms_box_corners` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static void nms_box_corners(const Tensor* boxes, int batch, int box_idx, int center_point_box,
                            double* y1, double* x1, double* y2, double* x2) {
    int num_boxes = boxes->shape[1];
    size_t base = ((size_t)batch * num_boxes + (size_t)box_idx) * 4;
    double a = get_value_as_double(boxes, base + 0);
    double b = get_value_as_double(boxes, base + 1);
    double c = get_value_as_double(boxes, base + 2);
    double d = get_value_as_double(boxes, base + 3);

    if (center_point_box) {
        double x_center = a;
        double y_center = b;
        double width = c;
        double height = d;
        *y1 = y_center - height / 2.0;
        *x1 = x_center - width / 2.0;
        *y2 = y_center + height / 2.0;
        *x2 = x_center + width / 2.0;
    } else {
        *y1 = a;
        *x1 = b;
        *y2 = c;
        *x2 = d;
    }

    if (*y1 > *y2) {
        double tmp = *y1;
        *y1 = *y2;
        *y2 = tmp;
    }
    if (*x1 > *x2) {
        double tmp = *x1;
        *x1 = *x2;
        *x2 = tmp;
    }
}

// 实现 `nms_iou` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double nms_iou(const Tensor* boxes, int batch, int lhs, int rhs, int center_point_box) {
    double ay1, ax1, ay2, ax2;
    double by1, bx1, by2, bx2;
    nms_box_corners(boxes, batch, lhs, center_point_box, &ay1, &ax1, &ay2, &ax2);
    nms_box_corners(boxes, batch, rhs, center_point_box, &by1, &bx1, &by2, &bx2);

    double inter_h = fmax(0.0, fmin(ay2, by2) - fmax(ay1, by1));
    double inter_w = fmax(0.0, fmin(ax2, bx2) - fmax(ax1, bx1));
    double inter = inter_h * inter_w;
    double area_a = fmax(0.0, ay2 - ay1) * fmax(0.0, ax2 - ax1);
    double area_b = fmax(0.0, by2 - by1) * fmax(0.0, bx2 - bx1);
    double union_area = area_a + area_b - inter;
    return union_area <= 0.0 ? 0.0 : inter / union_area;
}

// 实现 `non max suppression` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。

// 实现 `grid_denormalize` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_denormalize(double coord, int length, int align_corners) {
    if (align_corners) {
        return (coord + 1.0) * (double)(length - 1) / 2.0;
    }
    return ((coord + 1.0) * (double)length - 1.0) / 2.0;
}

// 实现 `grid_reflect_coordinate` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_reflect_coordinate(double coord, double low, double high) {
    if (high <= low) return low;
    double span = high - low;
    double value = fabs(fmod(coord - low, 2.0 * span));
    if (value > span) value = 2.0 * span - value;
    return value + low;
}

// 实现 `grid_sample_coordinate` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_sample_coordinate(double coord, int length, int padding_mode, int align_corners) {
    if (padding_mode == 1) {
        return fmin(fmax(coord, 0.0), (double)(length - 1));
    }
    if (padding_mode == 2) {
        double low = align_corners ? 0.0 : -0.5;
        double high = align_corners ? (double)(length - 1) : (double)length - 0.5;
        double reflected = grid_reflect_coordinate(coord, low, high);
        return fmin(fmax(reflected, 0.0), (double)(length - 1));
    }
    return coord;
}

// 实现 `grid_get_pixel_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_get_pixel_2d(const Tensor* input, int n, int c, double y, double x,
                                int padding_mode, int align_corners) {
    int height = input->shape[2];
    int width = input->shape[3];
    if (padding_mode == 1 || padding_mode == 2) {
        y = grid_sample_coordinate(y, height, padding_mode, align_corners);
        x = grid_sample_coordinate(x, width, padding_mode, align_corners);
    }
    int yi = (int)y;
    int xi = (int)x;
    if (yi < 0 || yi >= height || xi < 0 || xi >= width) return 0.0;
    size_t idx = ((size_t)n * input->shape[1] * height * width)
               + ((size_t)c * height * width)
               + ((size_t)yi * width)
               + (size_t)xi;
    return get_value_as_double(input, idx);
}

// 实现 `grid_bilinear_sample_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_bilinear_sample_2d(const Tensor* input, int n, int c, double y, double x,
                                      int padding_mode, int align_corners) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    double ly = y - (double)y0;
    double lx = x - (double)x0;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    return grid_get_pixel_2d(input, n, c, y0, x0, padding_mode, align_corners) * hy * hx
         + grid_get_pixel_2d(input, n, c, y0, x1, padding_mode, align_corners) * hy * lx
         + grid_get_pixel_2d(input, n, c, y1, x0, padding_mode, align_corners) * ly * hx
         + grid_get_pixel_2d(input, n, c, y1, x1, padding_mode, align_corners) * ly * lx;
}

// 实现 `grid_cubic_coefficients` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static void grid_cubic_coefficients(double t, double coeffs[4]) {
    double alpha = -0.75;
    double x = fabs(t);
    coeffs[0] = ((alpha * (x + 1.0) - 5.0 * alpha) * (x + 1.0) + 8.0 * alpha) * (x + 1.0) - 4.0 * alpha;
    coeffs[1] = ((alpha + 2.0) * x - (alpha + 3.0)) * x * x + 1.0;
    coeffs[2] = ((alpha + 2.0) * (1.0 - x) - (alpha + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    coeffs[3] = ((alpha * (2.0 - x) - 5.0 * alpha) * (2.0 - x) + 8.0 * alpha) * (2.0 - x) - 4.0 * alpha;
}

// 实现 `grid_bicubic_sample_2d` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double grid_bicubic_sample_2d(const Tensor* input, int n, int c, double y, double x,
                                     int padding_mode, int align_corners) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    double cy[4];
    double cx[4];
    grid_cubic_coefficients(y - (double)y0, cy);
    grid_cubic_coefficients(x - (double)x0, cx);
    double total = 0.0;
    for (int iy = 0; iy < 4; iy++) {
        for (int ix = 0; ix < 4; ix++) {
            total += cy[iy] * cx[ix] * grid_get_pixel_2d(
                input, n, c, y0 - 1 + iy, x0 - 1 + ix, padding_mode, align_corners
            );
        }
    }
    return total;
}

#endif /* TENSOR_OPS_INTERNAL_DETECTION_SAMPLING_H */
