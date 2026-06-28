/**
  ******************************************************************************
  * @file        tensor_ops_window.c
  * @author      Egor Izmaylov
  * @brief       实现窗口函数类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从 spectral shard 拆分 window 算子。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
// 实现 `hann window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hann_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return; // 甚至不需要写入
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = 0.5 * (1.0 - cos(2.0 * PI * i / denom));
        set_tensor_value_from_float(output, i, val);
    }
}


// Hamming Window: alpha - beta * cos(2*pi*n / (N-1)), alpha=25/46, beta=21/46
// 实现 `hamming window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hamming_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = (25.0 / 46.0) - (21.0 / 46.0) * cos(2.0 * PI * i / denom);
        set_tensor_value_from_float(output, i, val);
    }
}


// Blackman Window: 0.42 - 0.5*cos(...) + 0.08*cos(...)
// 实现 `blackman window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void blackman_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0); // center value usually
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double term1 = 0.5 * cos(2.0 * PI * i / denom);
        double term2 = 0.08 * cos(4.0 * PI * i / denom);
        double val = 0.42 - term1 + term2;
        set_tensor_value_from_float(output, i, val);
    }
}
