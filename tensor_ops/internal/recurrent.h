/**
  ******************************************************************************
  * @file        recurrent.h
  * @author      Egor Izmaylov
  * @brief       提供 RNN、GRU 和 LSTM 共用的激活、方向和布局索引辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_recurrent.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_RECURRENT_H
#define TENSOR_OPS_INTERNAL_RECURRENT_H

#include "../tensor_ops_internal.h"

/*
 * direction 编码保持 0=forward、1=reverse、2=bidirectional；layout=0 使用
 * [seq,batch,...]，layout=1 使用 [batch,seq,...]。索引 helper 集中维护这组约定。
 */

// 实现 `recurrent_alpha` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_alpha(const float* values, int index, double default_value) {
    if (!values) return default_value;
    float value = values[index];
    return isnan(value) ? default_value : (double)value;
}

// 实现 `recurrent_clip` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_clip(double value, float clip, int has_clip) {
    if (!has_clip) return value;
    if (value > (double)clip) return (double)clip;
    if (value < -(double)clip) return -(double)clip;
    return value;
}

// 实现 `recurrent_activation` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double recurrent_activation(double x, int code, const float* alphas, const float* betas, int index) {
    switch (code) {
        case 1:
            return 1.0 / (1.0 + exp(-x));
        case 2:
            return x > 0.0 ? x : 0.0;
        case 3: {
            double a = recurrent_alpha(alphas, index, 1.0);
            double b = recurrent_alpha(betas, index, 0.0);
            return a * x + b;
        }
        case 4: {
            double a = recurrent_alpha(alphas, index, 0.01);
            return x >= 0.0 ? x : a * x;
        }
        case 5: {
            double a = recurrent_alpha(alphas, index, 1.0);
            return x >= a ? x : 0.0;
        }
        case 6: {
            double a = recurrent_alpha(alphas, index, 1.0);
            double b = recurrent_alpha(betas, index, 1.0);
            return a * tanh(b * x);
        }
        case 7: {
            double a = recurrent_alpha(alphas, index, 0.2);
            double b = recurrent_alpha(betas, index, 0.5);
            double y = a * x + b;
            if (y < 0.0) return 0.0;
            if (y > 1.0) return 1.0;
            return y;
        }
        case 8: {
            double a = recurrent_alpha(alphas, index, 1.0);
            return x >= 0.0 ? x : a * (exp(x) - 1.0);
        }
        case 9:
            return x / (1.0 + fabs(x));
        case 10:
            return log1p(exp(x));
        case 0:
        default:
            return tanh(x);
    }
}

// 实现 `recurrent_activation_code` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_activation_code(const int* activations, int num_activations, int index, int default_code) {
    if (!activations || index >= num_activations) return default_code;
    return activations[index];
}

// 实现 `recurrent_num_dirs` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_num_dirs(int direction) {
    return direction == 2 ? 2 : 1;
}

// 实现 `recurrent_is_reverse` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_is_reverse(int direction, int dir_index) {
    return direction == 1 || (direction == 2 && dir_index == 1);
}

// 实现 `recurrent_x_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t recurrent_x_index(const Tensor* X, int layout, int t, int b, int i) {
    if (layout == 1) {
        int seq_len = X->shape[1];
        int input_size = X->shape[2];
        return ((size_t)b * seq_len * input_size) + ((size_t)t * input_size) + (size_t)i;
    }
    int batch = X->shape[1];
    int input_size = X->shape[2];
    return ((size_t)t * batch * input_size) + ((size_t)b * input_size) + (size_t)i;
}

// 实现 `recurrent_y_index` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static size_t recurrent_y_index(const Tensor* Y, int layout, int t, int d, int b, int h) {
    if (layout == 1) {
        int seq_len = Y->shape[1];
        int num_dirs = Y->shape[2];
        int hidden = Y->shape[3];
        return ((size_t)b * seq_len * num_dirs * hidden)
             + ((size_t)t * num_dirs * hidden)
             + ((size_t)d * hidden)
             + (size_t)h;
    }
    int num_dirs = Y->shape[1];
    int batch = Y->shape[2];
    int hidden = Y->shape[3];
    return ((size_t)t * num_dirs * batch * hidden)
         + ((size_t)d * batch * hidden)
         + ((size_t)b * hidden)
         + (size_t)h;
}

// 实现 `recurrent_sequence_active` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static int recurrent_sequence_active(const Tensor* sequence_lens, int t, int b) {
    if (!sequence_lens || !sequence_lens->data) return 1;
    return get_value_as_int64(sequence_lens, (size_t)b) > t;
}

#endif /* TENSOR_OPS_INTERNAL_RECURRENT_H */
