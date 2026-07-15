/**
  ******************************************************************************
  * @file        tensor_ops_matrix_quant.c
  * @author      Egor Izmaylov
  * @brief       实现矩阵乘、Gemm 和量化类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `gemm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gemm_forward(const Tensor* A, const Tensor* B, const Tensor* C, Tensor* Y, 
                  float alpha, float beta, int transA, int transB) {
    // 假设 A, B 已经是 2D 矩阵 (前端已处理 reshape)
    int M = (transA == 0) ? A->shape[0] : A->shape[1];
    int K = (transA == 0) ? A->shape[1] : A->shape[0];
    int N = (transB == 0) ? B->shape[1] : B->shape[0];
    
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            
            // 计算矩阵乘积: A' * B'
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                // 计算 A 的索引
                size_t idx_a = (transA == 0) ? ((size_t)m * A->shape[1] + k) 
                                             : ((size_t)k * A->shape[1] + m);
                
                // 计算 B 的索引
                size_t idx_b = (transB == 0) ? ((size_t)k * B->shape[1] + n) 
                                             : ((size_t)n * B->shape[1] + k);
                
                sum += get_value_as_double(A, idx_a) * get_value_as_double(B, idx_b);
            }
            
            double res = (double)alpha * sum;
            
            // 处理 Bias C
            if (C != NULL && C->data != NULL) {
                double val_c = 0.0;
                // 标量广播
                if (C->size == 1) {
                    val_c = get_value_as_double(C, 0);
                } 
                // 1D 张量处理 (通常是 (N,) 加在列上，或 (M,) 加在行上)
                else if (C->ndim == 1) {
                    if (C->shape[0] == N) {
                        val_c = get_value_as_double(C, n);
                    } 
                    else if (C->shape[0] == M) {
                        val_c = get_value_as_double(C, m);
                    }
                } 
                // 2D 及以上张量
                else if (C->ndim >= 2) {
                    int H = C->shape[C->ndim - 2]; // 倒数第二维
                    int W = C->shape[C->ndim - 1]; // 最后一维
                    int idx_h = (H == 1) ? 0 : m; 
                    int idx_w = (W == 1) ? 0 : n;

                    if (idx_h < H && idx_w < W) {
                        val_c = get_value_as_double(C, idx_h * W + idx_w);
                    }
                }
                res += (double)beta * val_c;
            }
            // 写入结果
            size_t y_idx = (size_t)m * N + n;
            set_tensor_value_from_float(Y, y_idx, res);
        }
    }
}


// 实现 `matmul integer` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void matmul_integer_forward(const Tensor* A, const Tensor* B,
                            const Tensor* AZeroPoint, const Tensor* BZeroPoint,
                            Tensor* Y) {
    if (!A || !B || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM || ndim < 2) return;

    int K = A->shape[A->ndim - 1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, Y->shape, ndim);

        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        int64_t sum = 0;

        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim;
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;
                else if (d == A->ndim - 2) val = m;
                else {
                    int y_dim_idx = d + offset_a;
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += (size_t)val * stride_a;
                stride_a *= A->shape[d];
            }

            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;
                else if (d == B->ndim - 2) val = k;
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += (size_t)val * stride_b;
                stride_b *= B->shape[d];
            }

            int64_t a_val = get_value_as_int64(A, idx_a);
            int64_t b_val = get_value_as_int64(B, idx_b);
            int64_t a_zp = (AZeroPoint && AZeroPoint->data) ? get_value_as_int64(AZeroPoint, idx_a) : 0;
            int64_t b_zp = (BZeroPoint && BZeroPoint->data) ? get_value_as_int64(BZeroPoint, idx_b) : 0;
            sum += (a_val - a_zp) * (b_val - b_zp);
        }

        if (Y->dtype == DTYPE_INT32) {
            ((int32_t*)Y->data)[i] = (int32_t)sum;
        } else {
            set_tensor_value_from_int(Y, i, sum);
        }
    }
}


// 实现 `qlinear matmul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void qlinear_matmul_forward(const Tensor* A, const Tensor* AScale, const Tensor* AZeroPoint,
                            const Tensor* B, const Tensor* BScale, const Tensor* BZeroPoint,
                            const Tensor* YScale, const Tensor* YZeroPoint, Tensor* Y) {
    if (!A || !AScale || !AZeroPoint || !B || !BScale || !BZeroPoint || !YScale || !YZeroPoint || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM || ndim < 2) return;

    int K = A->shape[A->ndim - 1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, Y->shape, ndim);

        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        double acc = 0.0;

        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim;
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;
                else if (d == A->ndim - 2) val = m;
                else {
                    int y_dim_idx = d + offset_a;
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += (size_t)val * stride_a;
                stride_a *= A->shape[d];
            }

            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;
                else if (d == B->ndim - 2) val = k;
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += (size_t)val * stride_b;
                stride_b *= B->shape[d];
            }

            double a_real = (get_value_as_double(A, idx_a) - get_value_as_double(AZeroPoint, idx_a)) * get_value_as_double(AScale, idx_a);
            double b_real = (get_value_as_double(B, idx_b) - get_value_as_double(BZeroPoint, idx_b)) * get_value_as_double(BScale, idx_b);
            acc += a_real * b_real;
        }

        double y_scale = get_value_as_double(YScale, i);
        double y_zp = get_value_as_double(YZeroPoint, i);
        double q = y_zp;
        if (y_scale != 0.0) {
            q = nearbyint(acc / y_scale + y_zp);
        }
        set_tensor_value_from_float(Y, i, q);
    }
}
