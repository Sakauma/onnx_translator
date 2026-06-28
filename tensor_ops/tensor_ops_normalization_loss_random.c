/**
  ******************************************************************************
  * @file        tensor_ops_normalization_loss_random.c
  * @author      Egor Izmaylov
  * @brief       实现归一化类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `lrn` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void lrn_forward(const Tensor* input, Tensor* output, int size, float alpha, float beta, float bias) {
    if (!input || !output || input->ndim < 3 || input->size != output->size || size <= 0) return;

    int channels = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    size_t batch_size = input->shape[0];
    int lower = (size - 1) / 2;
    int upper = size - 1 - lower;

    _Pragma("omp parallel for collapse(2)")
    for (size_t n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            int begin = c - lower;
            int end = c + upper + 1;
            if (begin < 0) begin = 0;
            if (end > channels) end = channels;

            for (size_t s = 0; s < spatial_size; s++) {
                double square_sum = 0.0;
                for (int cc = begin; cc < end; cc++) {
                    size_t idx = (n * (size_t)channels + (size_t)cc) * spatial_size + s;
                    double val = get_value_as_double(input, idx);
                    square_sum += val * val;
                }
                size_t out_idx = (n * (size_t)channels + (size_t)c) * spatial_size + s;
                double x = get_value_as_double(input, out_idx);
                double denom = pow((double)bias + ((double)alpha / (double)size) * square_sum, (double)beta);
                set_tensor_value_from_float(output, out_idx, x / denom);
            }
        }
    }
}
// 实现 `mean variance normalization` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mean_variance_normalization_forward(const Tensor* input, Tensor* output, ReduceParams* params) {
    if (!input || !output || !params || input->size != output->size) return;

    int ndim = input->ndim;
    int* axes = params->axes;
    int num_axes = params->num_axes;
    if (ndim > MAX_NDIM || num_axes < 1) return;

    size_t reduce_total_steps = 1;
    for (int i = 0; i < num_axes; i++) {
        if (axes[i] < 0 || axes[i] >= ndim) return;
        reduce_total_steps *= input->shape[axes[i]];
    }
    if (reduce_total_steps == 0) return;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        int base_coords[MAX_NDIM] = {0};
        get_coords_from_index(i, base_coords, input->shape, ndim);

        double sum = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            int coords[MAX_NDIM];
            memcpy(coords, base_coords, ndim * sizeof(int));
            size_t temp_r = r;
            for (int k = num_axes - 1; k >= 0; k--) {
                int axis_idx = axes[k];
                int dim_size = input->shape[axis_idx];
                coords[axis_idx] = temp_r % dim_size;
                temp_r /= dim_size;
            }
            size_t idx = get_index_from_coords(coords, input->shape, ndim);
            sum += get_value_as_double(input, idx);
        }

        double mean = sum / (double)reduce_total_steps;
        double sq_sum = 0.0;
        for (size_t r = 0; r < reduce_total_steps; r++) {
            int coords[MAX_NDIM];
            memcpy(coords, base_coords, ndim * sizeof(int));
            size_t temp_r = r;
            for (int k = num_axes - 1; k >= 0; k--) {
                int axis_idx = axes[k];
                int dim_size = input->shape[axis_idx];
                coords[axis_idx] = temp_r % dim_size;
                temp_r /= dim_size;
            }
            size_t idx = get_index_from_coords(coords, input->shape, ndim);
            double diff = get_value_as_double(input, idx) - mean;
            sq_sum += diff * diff;
        }

        double variance = sq_sum / (double)reduce_total_steps;
        double x = get_value_as_double(input, i);
        set_tensor_value_from_float(output, i, (x - mean) / sqrt(variance));
    }
}


// BatchNormalization (Inference Mode)
// Y = (X - mean) / sqrt(var + eps) * scale + B
// 优化为: Y = X * A + K
// 其中 A = scale / sqrt(var + eps), K = B - mean * A
// 实现 `batch norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void batch_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        const Tensor* mean, const Tensor* var, Tensor* output, float epsilon) {
    if (!input || !scale || !B || !mean || !var || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    // 假设输入是 NCHW 或 NC
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    // 预计算通道参数，避免在内层循环重复计算 sqrt/div
    double* A_table = (double*)malloc(C * sizeof(double));
    double* K_table = (double*)malloc(C * sizeof(double));
    
    for (int c = 0; c < C; c++) {
        double s = get_value_as_double(scale, c);
        double b = get_value_as_double(B, c);
        double m = get_value_as_double(mean, c);
        double v = get_value_as_double(var, c);
        
        double inv_std = 1.0 / sqrt(v + epsilon);
        A_table[c] = s * inv_std;
        K_table[c] = b - m * A_table[c];
    }
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            double A_val = A_table[c];
            double K_val = K_table[c];
            size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
            
            for (size_t i = 0; i < spatial_size; i++) {
                double x = get_value_as_double(input, offset + i);
                double y = x * A_val + K_val;
                set_tensor_value_from_float(output, offset + i, y);
            }
        }
    }
    
    free(A_table);
    free(K_table);
}


// BatchNormalization (Training Mode)
// 训练模式按通道从当前 batch 计算 saved mean/variance，并输出更新后的 running mean/variance。
// 实现 `batch norm training` 算子的 C 后端入口，保证训练态多输出不退回 Python 数值路径。
void batch_norm_training_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                 const Tensor* mean, const Tensor* var,
                                 Tensor* output, Tensor* running_mean, Tensor* running_var,
                                 float epsilon, float momentum) {
    if (!input || !scale || !B || !mean || !var || !output || !running_mean || !running_var) return;
    if (input->ndim < 2) return;

    int N = input->shape[0];
    int C = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= (size_t)input->shape[i];
    size_t sample_count = (size_t)N * spatial_size;
    if (N <= 0 || C <= 0 || sample_count == 0) return;

    #pragma omp parallel for
    for (int c = 0; c < C; c++) {
        double sum = 0.0;
        double sumsq = 0.0;
        for (int n = 0; n < N; n++) {
            size_t base = (size_t)n * (size_t)C * spatial_size + (size_t)c * spatial_size;
            for (size_t s = 0; s < spatial_size; s++) {
                double x = get_value_as_double(input, base + s);
                sum += x;
                sumsq += x * x;
            }
        }

        double saved_mean = sum / (double)sample_count;
        double saved_var = sumsq / (double)sample_count - saved_mean * saved_mean;
        if (saved_var < 0.0 && saved_var > -1e-12) saved_var = 0.0;

        double old_mean = get_value_as_double(mean, c);
        double old_var = get_value_as_double(var, c);
        double updated_mean = old_mean * (double)momentum + saved_mean * (1.0 - (double)momentum);
        double updated_var = old_var * (double)momentum + saved_var * (1.0 - (double)momentum);
        set_tensor_value_from_float(running_mean, c, updated_mean);
        set_tensor_value_from_float(running_var, c, updated_var);

        double s_val = get_value_as_double(scale, c);
        double b_val = get_value_as_double(B, c);
        double inv_std = 1.0 / sqrt(saved_var + (double)epsilon);
        for (int n = 0; n < N; n++) {
            size_t base = (size_t)n * (size_t)C * spatial_size + (size_t)c * spatial_size;
            for (size_t idx = 0; idx < spatial_size; idx++) {
                double x = get_value_as_double(input, base + idx);
                double y = s_val * (x - saved_mean) * inv_std + b_val;
                set_tensor_value_from_float(output, base + idx, y);
            }
        }
    }
}


// InstanceNormalization
// 对每个 (n, c) 切片计算均值和方差，然后归一化
// 实现 `instance norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void instance_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                           Tensor* output, float epsilon) {
    if (!input || !scale || !B || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
            
            double sum = 0.0;
            for (size_t i = 0; i < spatial_size; i++) {
                sum += get_value_as_double(input, offset + i);
            }
            double mean = sum / spatial_size;

            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < spatial_size; i++) {
                double val = get_value_as_double(input, offset + i);
                double diff = val - mean;
                sum_sq_diff += diff * diff;
            }
            double var = sum_sq_diff / spatial_size;
            double inv_std = 1.0 / sqrt(var + epsilon);
            
            double s = get_value_as_double(scale, c);
            double b = get_value_as_double(B, c);
            
            for (size_t i = 0; i < spatial_size; i++) {
                double x = get_value_as_double(input, offset + i);
                double y = (x - mean) * inv_std * s + b;
                set_tensor_value_from_float(output, offset + i, y);
            }
        }
    }
}


// LayerNormalization
// 沿着 axis 开始的后缀维度进行归一化。
// 实现 `layer norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void layer_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int axis, float epsilon) {
    if (!input || !output) return;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;
    
    size_t norm_dim = 1;
    for (int i = axis; i < ndim; i++) norm_dim *= (size_t)input->shape[i];
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= (size_t)input->shape[i];
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        size_t offset = i * norm_dim;
        
        double sum = 0.0;
        for (size_t j = 0; j < norm_dim; j++) {
            sum += get_value_as_double(input, offset + j);
        }
        double mean = sum / (double)norm_dim;
        
        double sum_sq_diff = 0.0;
        for (size_t j = 0; j < norm_dim; j++) {
            double val = get_value_as_double(input, offset + j);
            double diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        double var = sum_sq_diff / (double)norm_dim;
        double inv_std = 1.0 / sqrt(var + epsilon);
        
        for (size_t j = 0; j < norm_dim; j++) {
            double x = get_value_as_double(input, offset + j);
            
            double s = 1.0;
            double b = 0.0;
            if (scale) s = get_value_as_double(scale, j);
            if (B) b = get_value_as_double(B, j);
            
            double y = (x - mean) * inv_std * s + b;
            set_tensor_value_from_float(output, offset + j, y);
        }
    }
}


// LayerNormalization 多输出
// 沿着 axis 后缀维度归一化，同时输出每个归一化切片的 mean 和 inv_std。
// 实现 `layer norm` 多输出 C 后端入口，避免 mean/inv_std 辅助输出回退到 Python 数值路径。
void layer_norm_multi_output_forward(const Tensor* input, const Tensor* scale, const Tensor* B,
                                     Tensor* output, Tensor* mean_output, Tensor* inv_std_output,
                                     int axis, float epsilon) {
    if (!input || !output || !mean_output || !inv_std_output) return;

    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;

    size_t norm_dim = 1;
    for (int i = axis; i < ndim; i++) norm_dim *= (size_t)input->shape[i];
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= (size_t)input->shape[i];

    #pragma omp parallel for
    for (size_t row = 0; row < outer_size; row++) {
        size_t offset = row * norm_dim;

        double sum = 0.0;
        for (size_t col = 0; col < norm_dim; col++) {
            sum += get_value_as_double(input, offset + col);
        }
        double mean = sum / (double)norm_dim;

        double sum_sq_diff = 0.0;
        for (size_t col = 0; col < norm_dim; col++) {
            double value = get_value_as_double(input, offset + col);
            double diff = value - mean;
            sum_sq_diff += diff * diff;
        }
        double variance = sum_sq_diff / (double)norm_dim;
        double inv_std = 1.0 / sqrt(variance + (double)epsilon);

        set_tensor_value_from_float(mean_output, row, mean);
        set_tensor_value_from_float(inv_std_output, row, inv_std);

        for (size_t col = 0; col < norm_dim; col++) {
            double x = get_value_as_double(input, offset + col);
            double s = scale ? get_value_as_double(scale, col) : 1.0;
            double b = B ? get_value_as_double(B, col) : 0.0;
            double y = (x - mean) * inv_std * s + b;
            set_tensor_value_from_float(output, offset + col, y);
        }
    }
}
