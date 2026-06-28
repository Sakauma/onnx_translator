/**
  ******************************************************************************
  * @file        tensor_ops_recurrent.c
  * @author      Egor Izmaylov
  * @brief       实现循环网络类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `rnn` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void rnn_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 const int* activations, const float* activation_alpha,
                 const float* activation_beta, int num_activations,
                 float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    if (X->ndim != 3 || W->ndim != 3 || R->ndim != 3 || Y->ndim != 4) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[0] != num_dirs || R->shape[0] != num_dirs || W->shape[1] != hidden || R->shape[1] != hidden || R->shape[2] != hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* h_new = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !h_new) {
        free(h_state);
        free(h_new);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++) {
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    h_state[((size_t)d * batch + b) * hidden + h] =
                        get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
                }
            }
        }
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int act_code = recurrent_activation_code(activations, num_activations, d, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double pre = 0.0;
                    for (int i = 0; i < input_size; i++) {
                        pre += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                             * get_value_as_double(W, ((size_t)d * hidden + h) * input_size + i);
                    }
                    for (int hh = 0; hh < hidden; hh++) {
                        pre += h_state[((size_t)d * batch + b) * hidden + hh]
                             * get_value_as_double(R, ((size_t)d * hidden + h) * hidden + hh);
                    }
                    if (B && B->data) {
                        pre += get_value_as_double(B, (size_t)d * 2 * hidden + h);
                        pre += get_value_as_double(B, (size_t)d * 2 * hidden + hidden + h);
                    }
                    pre = recurrent_clip(pre, clip, has_clip);
                    h_new[(size_t)b * hidden + h] = recurrent_activation(pre, act_code, activation_alpha, activation_beta, d);
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    if (active) h_state[state_idx] = h_new[(size_t)b * hidden + h];
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    if (Y_h && Y_h->data) {
        for (int d = 0; d < num_dirs; d++) {
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    size_t idx = ((size_t)d * batch + b) * hidden + h;
                    set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                }
            }
        }
    }
    free(h_state);
    free(h_new);
}


// 实现 `gru` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gru_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                 const Tensor* sequence_lens, const Tensor* initial_h,
                 Tensor* Y, Tensor* Y_h, int hidden_size, int direction, int layout,
                 int linear_before_reset, const int* activations,
                 const float* activation_alpha, const float* activation_beta,
                 int num_activations, float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[1] != 3 * hidden || R->shape[1] != 3 * hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* z = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* reset = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* cand = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !z || !reset || !cand) {
        free(h_state); free(z); free(reset); free(cand);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    h_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int f_code = recurrent_activation_code(activations, num_activations, d * 2, 1);
        int g_code = recurrent_activation_code(activations, num_activations, d * 2 + 1, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double gate_pre[2] = {0.0, 0.0};
                    for (int gate = 0; gate < 2; gate++) {
                        for (int i = 0; i < input_size; i++) {
                            gate_pre[gate] += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                                * get_value_as_double(W, ((size_t)d * 3 * hidden + gate * hidden + h) * input_size + i);
                        }
                        for (int hh = 0; hh < hidden; hh++) {
                            gate_pre[gate] += h_state[((size_t)d * batch + b) * hidden + hh]
                                * get_value_as_double(R, ((size_t)d * 3 * hidden + gate * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            gate_pre[gate] += get_value_as_double(B, (size_t)d * 6 * hidden + gate * hidden + h);
                            gate_pre[gate] += get_value_as_double(B, (size_t)d * 6 * hidden + 3 * hidden + gate * hidden + h);
                        }
                        gate_pre[gate] = recurrent_clip(gate_pre[gate], clip, has_clip);
                    }
                    z[(size_t)b * hidden + h] = recurrent_activation(gate_pre[0], f_code, activation_alpha, activation_beta, d * 2);
                    reset[(size_t)b * hidden + h] = recurrent_activation(gate_pre[1], f_code, activation_alpha, activation_beta, d * 2);
                }
            }
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double pre = 0.0;
                    for (int i = 0; i < input_size; i++) {
                        pre += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                             * get_value_as_double(W, ((size_t)d * 3 * hidden + 2 * hidden + h) * input_size + i);
                    }
                    if (linear_before_reset) {
                        double rec = 0.0;
                        for (int hh = 0; hh < hidden; hh++) {
                            rec += h_state[((size_t)d * batch + b) * hidden + hh]
                                 * get_value_as_double(R, ((size_t)d * 3 * hidden + 2 * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) rec += get_value_as_double(B, (size_t)d * 6 * hidden + 5 * hidden + h);
                        pre += reset[(size_t)b * hidden + h] * rec;
                        if (B && B->data) pre += get_value_as_double(B, (size_t)d * 6 * hidden + 2 * hidden + h);
                    } else {
                        for (int hh = 0; hh < hidden; hh++) {
                            pre += reset[(size_t)b * hidden + hh] * h_state[((size_t)d * batch + b) * hidden + hh]
                                 * get_value_as_double(R, ((size_t)d * 3 * hidden + 2 * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            pre += get_value_as_double(B, (size_t)d * 6 * hidden + 2 * hidden + h);
                            pre += get_value_as_double(B, (size_t)d * 6 * hidden + 5 * hidden + h);
                        }
                    }
                    pre = recurrent_clip(pre, clip, has_clip);
                    cand[(size_t)b * hidden + h] = recurrent_activation(pre, g_code, activation_alpha, activation_beta, d * 2 + 1);
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    double h_old = h_state[state_idx];
                    double h_new = (1.0 - z[(size_t)b * hidden + h]) * cand[(size_t)b * hidden + h] + z[(size_t)b * hidden + h] * h_old;
                    if (active) h_state[state_idx] = h_new;
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    if (Y_h && Y_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++) {
                    size_t idx = ((size_t)d * batch + b) * hidden + h;
                    set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                }
    }
    free(h_state); free(z); free(reset); free(cand);
}


// 实现 `lstm` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void lstm_forward(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                  const Tensor* sequence_lens, const Tensor* initial_h,
                  const Tensor* initial_c, const Tensor* P,
                  Tensor* Y, Tensor* Y_h, Tensor* Y_c, int hidden_size,
                  int direction, int layout, int input_forget,
                  const int* activations, const float* activation_alpha,
                  const float* activation_beta, int num_activations,
                  float clip, int has_clip) {
    if (!X || !W || !R || !Y || !X->data || !W->data || !R->data || !Y->data) return;
    int seq_len = layout == 1 ? X->shape[1] : X->shape[0];
    int batch = layout == 1 ? X->shape[0] : X->shape[1];
    int input_size = X->shape[2];
    int num_dirs = recurrent_num_dirs(direction);
    int hidden = hidden_size > 0 ? hidden_size : R->shape[2];
    if (W->shape[1] != 4 * hidden || R->shape[1] != 4 * hidden) return;

    double* h_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* c_state = (double*)calloc((size_t)num_dirs * batch * hidden, sizeof(double));
    double* h_next = (double*)calloc((size_t)batch * hidden, sizeof(double));
    double* c_next = (double*)calloc((size_t)batch * hidden, sizeof(double));
    if (!h_state || !c_state || !h_next || !c_next) {
        free(h_state); free(c_state); free(h_next); free(c_next);
        return;
    }
    if (initial_h && initial_h->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    h_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_h, ((size_t)d * batch + b) * hidden + h);
    }
    if (initial_c && initial_c->data) {
        for (int d = 0; d < num_dirs; d++)
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                    c_state[((size_t)d * batch + b) * hidden + h] = get_value_as_double(initial_c, ((size_t)d * batch + b) * hidden + h);
    }

    for (int d = 0; d < num_dirs; d++) {
        int reverse = recurrent_is_reverse(direction, d);
        int f_code = recurrent_activation_code(activations, num_activations, d * 3, 1);
        int g_code = recurrent_activation_code(activations, num_activations, d * 3 + 1, 0);
        int h_code = recurrent_activation_code(activations, num_activations, d * 3 + 2, 0);
        for (int step = 0; step < seq_len; step++) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < hidden; h++) {
                    double gates[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int gate = 0; gate < 4; gate++) {
                        for (int i = 0; i < input_size; i++) {
                            gates[gate] += get_value_as_double(X, recurrent_x_index(X, layout, t, b, i))
                                * get_value_as_double(W, ((size_t)d * 4 * hidden + gate * hidden + h) * input_size + i);
                        }
                        for (int hh = 0; hh < hidden; hh++) {
                            gates[gate] += h_state[((size_t)d * batch + b) * hidden + hh]
                                * get_value_as_double(R, ((size_t)d * 4 * hidden + gate * hidden + h) * hidden + hh);
                        }
                        if (B && B->data) {
                            gates[gate] += get_value_as_double(B, (size_t)d * 8 * hidden + gate * hidden + h);
                            gates[gate] += get_value_as_double(B, (size_t)d * 8 * hidden + 4 * hidden + gate * hidden + h);
                        }
                    }
                    double c_prev = c_state[((size_t)d * batch + b) * hidden + h];
                    double p_i = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + h) : 0.0;
                    double p_o = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + hidden + h) : 0.0;
                    double p_f = (P && P->data) ? get_value_as_double(P, (size_t)d * 3 * hidden + 2 * hidden + h) : 0.0;
                    double i_gate = recurrent_activation(recurrent_clip(gates[0] + p_i * c_prev, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    double f_gate = input_forget ? (1.0 - i_gate) : recurrent_activation(recurrent_clip(gates[2] + p_f * c_prev, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    double c_bar = recurrent_activation(recurrent_clip(gates[3], clip, has_clip), g_code, activation_alpha, activation_beta, d * 3 + 1);
                    double c_val = f_gate * c_prev + i_gate * c_bar;
                    double o_gate = recurrent_activation(recurrent_clip(gates[1] + p_o * c_val, clip, has_clip), f_code, activation_alpha, activation_beta, d * 3);
                    h_next[(size_t)b * hidden + h] = o_gate * recurrent_activation(c_val, h_code, activation_alpha, activation_beta, d * 3 + 2);
                    c_next[(size_t)b * hidden + h] = c_val;
                }
            }
            for (int b = 0; b < batch; b++) {
                int active = recurrent_sequence_active(sequence_lens, t, b);
                for (int h = 0; h < hidden; h++) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    if (active) {
                        h_state[state_idx] = h_next[(size_t)b * hidden + h];
                        c_state[state_idx] = c_next[(size_t)b * hidden + h];
                    }
                    set_tensor_value_from_float(Y, recurrent_y_index(Y, layout, t, d, b, h), h_state[state_idx]);
                }
            }
        }
    }

    for (int d = 0; d < num_dirs; d++)
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < hidden; h++) {
                size_t idx = ((size_t)d * batch + b) * hidden + h;
                if (Y_h && Y_h->data) set_tensor_value_from_float(Y_h, idx, h_state[idx]);
                if (Y_c && Y_c->data) set_tensor_value_from_float(Y_c, idx, c_state[idx]);
            }
    free(h_state); free(c_state); free(h_next); free(c_next);
}
