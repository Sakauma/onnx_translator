/**
  ******************************************************************************
  * @file        tensor_ops_matmul.c
  * @author      Egor Izmaylov
  * @brief       实现 MatMul 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从 matrix/quant shard 拆分 MatMul。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// MatMul 实现 (无加速)
// 实现 `matmul` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void matmul_forward(const Tensor* A, const Tensor* B, Tensor* Y) {
    if (!A || !B || !Y) return;
    int ndim = Y->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }
    if (ndim < 2) return; // 至少是 2D
    int K = A->shape[A->ndim - 1];
    int use_integer_path = is_integer_dtype(A->dtype) && is_integer_dtype(B->dtype) && is_integer_dtype(Y->dtype);
    #pragma omp parallel for
    for (size_t i = 0; i < Y->size; i++) {
        int coords[MAX_NDIM] = {0}; // 最大 16 维
        get_coords_from_index(i, coords, Y->shape, ndim);
        // 当前计算的是 Y[..., m, n]
        int m = coords[ndim - 2];
        int n = coords[ndim - 1];
        double sum = 0.0;
        uint64_t integer_sum = 0;
        // 内积循环 K
        for (int k = 0; k < K; k++) {
            size_t idx_a = 0;
            size_t stride_a = 1;
            int offset_a = ndim - A->ndim; // 维度对齐偏移量
            for (int d = A->ndim - 1; d >= 0; d--) {
                int val;
                if (d == A->ndim - 1) val = k;       // 最后一维 K
                else if (d == A->ndim - 2) val = m;  // 倒数第二维 M
                else {
                    // Batch 维
                    int y_dim_idx = d + offset_a;
                    // 如果 A 在此维是 1，则广播取 0；否则跟随 Y 的坐标
                    val = (A->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_a += val * stride_a;
                stride_a *= A->shape[d];
            }
            // 计算 B 的索引 (逻辑同上)
            size_t idx_b = 0;
            size_t stride_b = 1;
            int offset_b = ndim - B->ndim;
            for (int d = B->ndim - 1; d >= 0; d--) {
                int val;
                if (d == B->ndim - 1) val = n;       // 最后一维 N
                else if (d == B->ndim - 2) val = k;  // 倒数第二维 K
                else {
                    int y_dim_idx = d + offset_b;
                    val = (B->shape[d] == 1) ? 0 : coords[y_dim_idx];
                }
                idx_b += val * stride_b;
                stride_b *= B->shape[d];
            }
            if (use_integer_path) {
                // 整数 MatMul 按目标 dtype 的二补码位宽自然回绕，避免 double 丢失 int64/uint64 低位。
                uint64_t val_a = get_integer_value_as_uint64(A, idx_a);
                uint64_t val_b = get_integer_value_as_uint64(B, idx_b);
                integer_sum += val_a * val_b;
            } else {
                // 混合精度浮点计算核心：float16/bfloat16/float8 先提升到 double 再累加。
                double val_a = get_value_as_double(A, idx_a);
                double val_b = get_value_as_double(B, idx_b);
                sum += val_a * val_b;
            }
        }
        // 结果存回
        if (use_integer_path) {
            set_integer_value_wrapped(Y, i, integer_sum);
        } else {
            set_tensor_value_from_float(Y, i, sum);
        }
    }
}
