/**
  ******************************************************************************
  * @file        tensor_ops_normalization_loss_random.c
  * @author      Egor Izmaylov
  * @brief       实现归一化、损失、随机和采样类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 根据元素下标派生随机状态，保证随机算子的输出不受 OpenMP 线程数和调度顺序影响。
static uint32_t random_state_for_index(uint32_t base_seed, size_t index) {
    return base_seed ^ (uint32_t)index;
}


// 生成与 CUDA verifier 一致的 [0, 1) 均匀随机数。
static double random_uniform01_for_index(uint32_t base_seed, size_t index) {
    uint32_t state = random_state_for_index(base_seed, index);
    uint32_t r = simple_lcg(&state);
    return (double)r / 2147483648.0;
}


// 使用 Box-Muller 变换生成与 CUDA verifier 一致的标准正态随机数。
static double random_normal01_for_index(uint32_t base_seed, size_t index) {
    uint32_t state = random_state_for_index(base_seed, index);
    uint32_t r1 = simple_lcg(&state);
    uint32_t r2 = simple_lcg(&state);
    double u1 = ((double)r1 + 1.0) / 2147483649.0;
    double u2 = ((double)r2 + 1.0) / 2147483649.0;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}


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


// 实现 `multinomial` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void multinomial_forward(const Tensor* input, Tensor* output, int sample_size, uint32_t seed) {
    if (!input || !output || input->ndim != 2 || output->ndim != 2 || sample_size < 0) return;
    int batch = input->shape[0];
    int classes = input->shape[1];
    if (output->shape[0] != batch || output->shape[1] != sample_size) return;

    for (int row = 0; row < batch; row++) {
        double total = 0.0;
        for (int c = 0; c < classes; c++) {
            double p = get_value_as_double(input, (size_t)row * classes + c);
            if (p > 0.0) total += p;
        }
        if (total <= 0.0) continue;

        uint32_t state = seed ? (seed + (uint32_t)row * 747796405u) : (uint32_t)time(NULL) + (uint32_t)row;
        for (int sample = 0; sample < sample_size; sample++) {
            uint32_t r = simple_lcg(&state);
            double threshold = ((double)r / 2147483648.0) * total;
            double cumulative = 0.0;
            int selected = classes - 1;
            for (int c = 0; c < classes; c++) {
                double p = get_value_as_double(input, (size_t)row * classes + c);
                if (p <= 0.0) continue;
                cumulative += p;
                if (threshold < cumulative) {
                    selected = c;
                    break;
                }
            }
            set_tensor_value_from_int(output, (size_t)row * sample_size + sample, selected);
        }
    }
}


// 实现 `negative log likelihood loss` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void negative_log_likelihood_loss_forward(const Tensor* input, const Tensor* target, const Tensor* weight,
                                          Tensor* output, int reduction, int has_ignore_index, int64_t ignore_index) {
    if (!input || !target || !output || input->ndim < 2) return;
    int batch = input->shape[0];
    int classes = input->shape[1];
    size_t spatial = loss_spatial_size(input);
    size_t total = (size_t)batch * spatial;
    double sum = 0.0;
    double denom = 0.0;

    for (size_t i = 0; i < total; i++) {
        int64_t cls = get_value_as_int64(target, i);
        double weighted_loss = 0.0;
        double cur_weight = 0.0;
        if (!(has_ignore_index && cls == ignore_index) && cls >= 0 && cls < classes) {
            cur_weight = loss_target_weight(weight, cls);
            size_t n = i / spatial;
            size_t s = i % spatial;
            size_t input_idx = n * (size_t)classes * spatial + (size_t)cls * spatial + s;
            weighted_loss = -get_value_as_double(input, input_idx) * cur_weight;
        }

        if (reduction == 0) {
            set_tensor_value_from_float(output, i, weighted_loss);
        } else {
            sum += weighted_loss;
            if (weight || has_ignore_index) denom += cur_weight;
            else denom += 1.0;
        }
    }

    if (reduction == 2) {
        set_tensor_value_from_float(output, 0, sum);
    } else if (reduction == 1) {
        set_tensor_value_from_float(output, 0, denom == 0.0 ? NAN : sum / denom);
    }
}


// 实现 `softmax cross entropy loss` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void softmax_cross_entropy_loss_forward(const Tensor* scores, const Tensor* labels, const Tensor* weights,
                                        Tensor* loss_output, Tensor* log_prob_output,
                                        int reduction, int has_ignore_index, int64_t ignore_index) {
    if (!scores || !labels || !loss_output || scores->ndim < 2) return;
    int batch = scores->shape[0];
    int classes = scores->shape[1];
    size_t spatial = loss_spatial_size(scores);
    double loss_sum = 0.0;
    double denom = 0.0;

    for (size_t n = 0; n < (size_t)batch; n++) {
        for (size_t s = 0; s < spatial; s++) {
            double max_val = -INFINITY;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                double value = get_value_as_double(scores, idx);
                if (value > max_val) max_val = value;
            }

            double exp_sum = 0.0;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                exp_sum += exp(get_value_as_double(scores, idx) - max_val);
            }
            double log_sum = log(exp_sum);

            size_t flat_target = n * spatial + s;
            int64_t cls = get_value_as_int64(labels, flat_target);
            double selected_loss = 0.0;
            double cur_weight = 0.0;
            for (int c = 0; c < classes; c++) {
                size_t idx = n * (size_t)classes * spatial + (size_t)c * spatial + s;
                double log_prob = get_value_as_double(scores, idx) - max_val - log_sum;
                if (log_prob_output) set_tensor_value_from_float(log_prob_output, idx, log_prob);
                if (c == cls && !(has_ignore_index && cls == ignore_index)) {
                    cur_weight = loss_target_weight(weights, cls);
                    selected_loss = -log_prob * cur_weight;
                }
            }

            if (reduction == 0) {
                set_tensor_value_from_float(loss_output, flat_target, selected_loss);
            } else {
                loss_sum += selected_loss;
                if (!(has_ignore_index && cls == ignore_index)) {
                    if (weights) denom += cur_weight;
                    else denom += 1.0;
                }
            }
        }
    }

    if (reduction == 2) {
        set_tensor_value_from_float(loss_output, 0, loss_sum);
    } else if (reduction == 1) {
        set_tensor_value_from_float(loss_output, 0, denom == 0.0 ? NAN : loss_sum / denom);
    }
}


// 实现 `non max suppression` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
int non_max_suppression_forward(const Tensor* boxes, const Tensor* scores, Tensor* output,
                                int max_output_boxes_per_class, float iou_threshold,
                                float score_threshold, int center_point_box) {
    if (!boxes || !scores || !output) return 0;
    if (boxes->ndim != 3 || scores->ndim != 3 || boxes->shape[2] != 4) return 0;
    int batch_count = boxes->shape[0];
    int num_boxes = boxes->shape[1];
    int class_count = scores->shape[1];
    if (scores->shape[0] != batch_count || scores->shape[2] != num_boxes || max_output_boxes_per_class <= 0) return 0;

    int* candidates = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    int* kept = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    if (!candidates || !kept) {
        free(candidates);
        free(kept);
        return 0;
    }

    int out_rows = 0;
    for (int b = 0; b < batch_count; b++) {
        for (int cls = 0; cls < class_count; cls++) {
            int candidate_count = 0;
            for (int box = 0; box < num_boxes; box++) {
                size_t score_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)box;
                double score = get_value_as_double(scores, score_idx);
                if (score >= (double)score_threshold) {
                    candidates[candidate_count++] = box;
                }
            }

            for (int i = 1; i < candidate_count; i++) {
                int current = candidates[i];
                size_t current_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)current;
                double current_score = get_value_as_double(scores, current_idx);
                int j = i - 1;
                while (j >= 0) {
                    int prev = candidates[j];
                    size_t prev_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)prev;
                    double prev_score = get_value_as_double(scores, prev_idx);
                    if (prev_score >= current_score) break;
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = current;
            }

            int kept_count = 0;
            for (int i = 0; i < candidate_count && kept_count < max_output_boxes_per_class; i++) {
                int candidate = candidates[i];
                int suppress = 0;
                for (int k = 0; k < kept_count; k++) {
                    if (nms_iou(boxes, b, candidate, kept[k], center_point_box) > (double)iou_threshold) {
                        suppress = 1;
                        break;
                    }
                }
                if (!suppress) {
                    kept[kept_count++] = candidate;
                    if ((size_t)(out_rows + 1) * 3 <= output->size) {
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 0, b);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 1, cls);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 2, candidate);
                    }
                    out_rows++;
                }
            }
        }
    }

    free(candidates);
    free(kept);
    return out_rows;
}


// 实现 `grid sample` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void grid_sample_forward(const Tensor* input, const Tensor* grid, Tensor* output,
                         int mode, int padding_mode, int align_corners) {
    if (!input || !grid || !output) return;
    if (input->ndim != 4 || grid->ndim != 4 || output->ndim != 4 || grid->shape[3] != 2) return;
    int n_batches = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];
    int out_h = grid->shape[1];
    int out_w = grid->shape[2];
    if (grid->shape[0] != n_batches || output->shape[0] != n_batches || output->shape[1] != channels ||
        output->shape[2] != out_h || output->shape[3] != out_w) return;

    _Pragma("omp parallel for collapse(4)")
    for (int n = 0; n < n_batches; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    size_t grid_idx = ((size_t)n * out_h * out_w * 2) + ((size_t)oy * out_w * 2) + ((size_t)ox * 2);
                    double x_norm = get_value_as_double(grid, grid_idx);
                    double y_norm = get_value_as_double(grid, grid_idx + 1);
                    double in_x = grid_denormalize(x_norm, width, align_corners);
                    double in_y = grid_denormalize(y_norm, height, align_corners);
                    double value;
                    if (mode == 1) {
                        double sy = nearbyint(grid_sample_coordinate(in_y, height, padding_mode, align_corners));
                        double sx = nearbyint(grid_sample_coordinate(in_x, width, padding_mode, align_corners));
                        value = grid_get_pixel_2d(input, n, c, sy, sx, padding_mode, align_corners);
                    } else if (mode == 2) {
                        value = grid_bicubic_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    } else {
                        value = grid_bilinear_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    }
                    size_t out_idx = ((size_t)n * channels * out_h * out_w)
                                   + ((size_t)c * out_h * out_w)
                                   + ((size_t)oy * out_w)
                                   + (size_t)ox;
                    set_tensor_value_from_float(output, out_idx, value);
                }
            }
        }
    }
}


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


// 实现 `random uniform like` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void random_uniform_like_forward(Tensor* output, float low, float high, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    double range = high - low;

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double r_norm = random_uniform01_for_index(base_seed, i);
        double val = low + r_norm * range;
        set_tensor_value_from_float(output, i, val);
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


// 根据输出元素坐标计算 scale 在单向广播规则下对应的元素索引。
static size_t rms_scale_broadcast_index(const Tensor* input, const Tensor* scale, size_t output_index) {
    if (!input || !scale || !scale->data || scale->size == 0) return (size_t)-1;
    if (scale->ndim == 0 || scale->size == 1) return 0;
    if (scale->ndim > input->ndim || input->ndim > MAX_NDIM || scale->ndim > MAX_NDIM) return (size_t)-1;

    int input_coords[MAX_NDIM] = {0};
    int scale_coords[MAX_NDIM] = {0};
    get_coords_from_index(output_index, input_coords, input->shape, input->ndim);

    int offset = input->ndim - scale->ndim;
    for (int i = 0; i < scale->ndim; i++) {
        int input_axis = offset + i;
        int scale_dim = scale->shape[i];
        if (scale_dim == 1) {
            scale_coords[i] = 0;
        } else if (scale_dim == input->shape[input_axis]) {
            scale_coords[i] = input_coords[input_axis];
        } else {
            return (size_t)-1;
        }
    }

    return get_index_from_coords(scale_coords, scale->shape, scale->ndim);
}


// 实现 `rms normalization` 算子的 C 后端入口，按 axis 后缀计算 RMS 并应用 scale 广播。
void rms_normalization_forward(const Tensor* input, const Tensor* scale, Tensor* output,
                               int axis, float epsilon, int stash_type) {
    if (!input || !scale || !output || input->size != output->size) return;
    if (input->ndim <= 0 || input->ndim > MAX_NDIM) return;

    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;
    if (scale->ndim > ndim || scale->ndim > MAX_NDIM) return;

    size_t normalized_size = 1;
    for (int i = axis; i < ndim; i++) normalized_size *= (size_t)input->shape[i];
    if (normalized_size == 0) return;
    size_t row_count = input->size / normalized_size;
    int use_double_stash = (stash_type == 11);

    #pragma omp parallel for
    for (size_t row = 0; row < row_count; row++) {
        size_t row_offset = row * normalized_size;
        if (use_double_stash) {
            double square_sum = 0.0;
            for (size_t j = 0; j < normalized_size; j++) {
                double x = get_value_as_double(input, row_offset + j);
                square_sum += x * x;
            }
            double inv_rms = 1.0 / sqrt(square_sum / (double)normalized_size + (double)epsilon);
            for (size_t j = 0; j < normalized_size; j++) {
                size_t out_idx = row_offset + j;
                size_t scale_idx = rms_scale_broadcast_index(input, scale, out_idx);
                if (scale_idx == (size_t)-1) continue;
                double x = get_value_as_double(input, out_idx);
                double s = get_value_as_double(scale, scale_idx);
                set_tensor_value_from_float(output, out_idx, x * inv_rms * s);
            }
        } else {
            float square_sum = 0.0f;
            for (size_t j = 0; j < normalized_size; j++) {
                float x = (float)get_value_as_double(input, row_offset + j);
                square_sum += x * x;
            }
            float inv_rms = 1.0f / sqrtf(square_sum / (float)normalized_size + epsilon);
            for (size_t j = 0; j < normalized_size; j++) {
                size_t out_idx = row_offset + j;
                size_t scale_idx = rms_scale_broadcast_index(input, scale, out_idx);
                if (scale_idx == (size_t)-1) continue;
                float x = (float)get_value_as_double(input, out_idx);
                float s = (float)get_value_as_double(scale, scale_idx);
                set_tensor_value_from_float(output, out_idx, (double)(x * inv_rms * s));
            }
        }
    }
}


// RandomNormal: Box-Muller 变换
// 实现 `random normal` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void random_normal_forward(Tensor* output, float mean, float scale, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double z0 = random_normal01_for_index(base_seed, i);
        double val = (double)mean + z0 * (double)scale;
        set_tensor_value_from_float(output, i, val);
    }
}


// Bernoulli: 生成 0 或 1
// 实现 `bernoulli` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bernoulli_forward(const Tensor* input, Tensor* output, float seed) {
    if (!input || !output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double prob = get_value_as_double(input, i);
        double r_norm = random_uniform01_for_index(base_seed, i);
        double res = (r_norm < prob) ? 1.0 : 0.0;
        set_tensor_value_from_float(output, i, res);
    }
}


// Dropout (Inference Mode)
// 实现 `dropout` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dropout_forward(const Tensor* input, Tensor* output, float ratio, int training_mode) {
    if (!input || !output) return;
    
    // 如果是推理模式(training_mode=0)，或者是比例为0，直接复制
    if (training_mode == 0 || ratio == 0.0f) {
        size_t elem_size = get_dtype_size(input->dtype);
        // 如果输入输出类型一致且大小一致
        if (input->dtype == output->dtype && input->size == output->size) {
            memcpy(output->data, input->data, input->size * elem_size);
        } else {
            // 类型转换复制
            cast_forward(input, output);
        }
        return;
    }
    
    // 训练模式下的 Dropout (简单的随机置0)
    // 标准 Dropout 还需要 scale (val / (1-ratio)) 以保持期望值
    double scale_factor = 1.0 / (1.0 - (double)ratio);
    uint32_t base_seed = (uint32_t)time(NULL);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t local_state = base_seed + tid;
        
        #pragma omp for
        for (size_t i = 0; i < input->size; i++) {
            uint32_t r = simple_lcg(&local_state);
            double r_norm = (double)r / 2147483648.0;
            
            double val = get_value_as_double(input, i);
            if (r_norm < ratio) {
                set_tensor_value_from_float(output, i, 0.0);
            } else {
                set_tensor_value_from_float(output, i, val * scale_factor);
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


// GroupNormalization
// 实现 `group norm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void group_norm_forward(const Tensor* input, const Tensor* scale, const Tensor* B, 
                        Tensor* output, int num_groups, float epsilon) {
    if (!input || !scale || !B || !output) return;
    
    int N = input->shape[0];
    int C = input->shape[1];
    
    // 检查能否整除
    if (C % num_groups != 0) return;
    int channels_per_group = C / num_groups;
    
    // 计算空间大小 (H * W * ...)
    size_t spatial_size = 1;
    for (int i = 2; i < input->ndim; i++) spatial_size *= input->shape[i];
    
    // 每个 Group 的元素数量
    size_t group_size = channels_per_group * spatial_size;
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            // 计算当前 Group 的 Mean 和 Var
            // Group 的数据范围：从 channel_start 到 channel_end
            int c_start = g * channels_per_group;
            int c_end = c_start + channels_per_group;
            
            double sum = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    sum += get_value_as_double(input, offset + i);
                }
            }
            double mean = sum / group_size;
            
            double sum_sq_diff = 0.0;
            for (int c = c_start; c < c_end; c++) {
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double val = get_value_as_double(input, offset + i);
                    double diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
            }
            double var = sum_sq_diff / group_size;
            double inv_std = 1.0 / sqrt(var + epsilon);
            
            // 应用归一化和仿射变换
            for (int c = c_start; c < c_end; c++) {
                double s_val = get_value_as_double(scale, c);
                double b_val = get_value_as_double(B, c);

                double A = inv_std * s_val;
                double K = b_val - mean * A;
                
                size_t offset = (size_t)n * C * spatial_size + (size_t)c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    double x = get_value_as_double(input, offset + i);
                    double y = x * A + K;
                    set_tensor_value_from_float(output, offset + i, y);
                }
            }
        }
    }
}
