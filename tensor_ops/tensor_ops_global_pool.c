/**
  ******************************************************************************
  * @file        tensor_ops_global_pool.c
  * @author      Egor Izmaylov
  * @brief       实现全局池化类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// GlobalAveragePool
// 假设输入是 NCHW (或至少后两维是空间维度)，如果不符合则不执行
// 实现 `global average pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_average_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;

    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];

    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            sum += get_value_as_double(input, offset + i);
        }
        set_tensor_value_from_float(output, n, sum / spatial_size);
    }
}


// GlobalMaxPool
// 实现 `global max pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_max_pool_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;

    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];

    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double max_val = -DBL_MAX;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            if (val > max_val) max_val = val;
        }
        set_tensor_value_from_float(output, n, max_val);
    }
}


// GlobalLpPool
// 实现 `global lp pool` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void global_lp_pool_forward(const Tensor* input, Tensor* output, int p) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return;

    size_t outer_size = (size_t)input->shape[0] * (size_t)input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < ndim; i++) spatial_size *= input->shape[i];

    _Pragma("omp parallel for")
    for (size_t n = 0; n < outer_size; n++) {
        double sum_pow = 0.0;
        size_t offset = n * spatial_size;
        for (size_t i = 0; i < spatial_size; i++) {
            double val = get_value_as_double(input, offset + i);
            sum_pow += pow(fabs(val), p);
        }

        // p=1 时就是 Sum(|x|)，p=2 时是 L2 Norm，p=inf 时是 Max
        double res = pow(sum_pow, 1.0 / p);
        set_tensor_value_from_float(output, n, res);
    }
}
