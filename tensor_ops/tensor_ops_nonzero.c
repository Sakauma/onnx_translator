/**
  ******************************************************************************
  * @file        tensor_ops_nonzero.c
  * @author      Egor Izmaylov
  * @brief       实现 NonZero C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// NonZero
// 实现 `nonzero` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void nonzero_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;

    int ndim = input->ndim;
    int64_t* out_ptr = (int64_t*)output->data; // NonZero 输出必定是 int64

    size_t current_col = 0;
    int coords[MAX_NDIM];

    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i);
        if (val != 0.0) {
            get_coords_from_index(i, coords, input->shape, ndim);
            // 写入 Output: Output 是 [ndim, N] 的矩阵
            // 转置存储：col 对应第 n 个非零元素，row 对应维度
            for (int d = 0; d < ndim; d++) {
                // index = d * N + current_col
                out_ptr[d * (output->shape[1]) + current_col] = (int64_t)coords[d];
            }
            current_col++;
        }
    }
}
