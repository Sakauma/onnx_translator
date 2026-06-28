/**
  ******************************************************************************
  * @file        tensor_ops_softmax_family.c
  * @author      Egor Izmaylov
  * @brief       实现 Softmax、Hardmax、LogSoftmax 和 LpNormalization 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// ================== Softmax 实现 ==================
// 实现 `softmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void softmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (axis < 0) axis += input->ndim;
    
    // 将 Tensor 视为 [Outer, Inner, Remaining]
    int inner_dim = input->shape[axis];
    
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) max_val = val;
            }
            double sum = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum += exp(val - max_val);
            }
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + 
                             (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                double res = exp(val - max_val) / sum;
                set_tensor_value_from_float(output, idx, res);
            }
        }
    }
}


// Hardmax
// 实现 `hardmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hardmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            int max_idx = 0;
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double res = (j == max_idx) ? 1.0 : 0.0;
                set_tensor_value_from_float(output, idx, res);
            }
        }
    }
}


// LogSoftmax: x - max - log(sum(exp(x - max)))
// 实现 `log softmax` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void log_softmax_forward(const Tensor* input, Tensor* output, int axis) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double max_val = -DBL_MAX;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                if (val > max_val) max_val = val;
            }
            
            double sum_exp = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum_exp += exp(val - max_val);
            }
            double log_sum = log(sum_exp);
            
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                double res = (val - max_val) - log_sum;
                set_tensor_value_from_float(output, idx, res);
            }
        }
    }
}


// LpNormalization
// y = x / ||x||_p
// 实现 `lp normalization` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void lp_normalization_forward(const Tensor* input, Tensor* output, int axis, int p) {
    if (!input || !output) return;
    if (axis < 0) axis += input->ndim;
    
    int inner_dim = input->shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) outer_dim *= input->shape[i];
    int remaining_dim = 1;
    for (int i = axis + 1; i < input->ndim; i++) remaining_dim *= input->shape[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_dim; i++) {
        for (int k = 0; k < remaining_dim; k++) {
            
            double sum_pow = 0.0;
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                sum_pow += pow(fabs(val), p);
            }
            
            double norm = pow(sum_pow, 1.0 / p);
            for (int j = 0; j < inner_dim; j++) {
                size_t idx = (size_t)i * inner_dim * remaining_dim + (size_t)j * remaining_dim + k;
                double val = get_value_as_double(input, idx);
                set_tensor_value_from_float(output, idx, norm == 0.0 ? 0.0 : val / norm);
            }
        }
    }
}
