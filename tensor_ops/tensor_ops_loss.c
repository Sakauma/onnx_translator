/**
  ******************************************************************************
  * @file        tensor_ops_loss.c
  * @author      Egor Izmaylov
  * @brief       实现损失函数类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

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
