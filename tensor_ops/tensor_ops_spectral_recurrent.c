/*
 * 文件功能：实现谱分析、窗口函数和循环网络类 C 后端算子。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include "tensor_ops_internal.h"


// 实现 `det` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void det_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->ndim < 2) return;

    int n = input->shape[input->ndim - 1];
    int m = input->shape[input->ndim - 2];
    if (n != m || n <= 0) return;

    size_t matrix_size = (size_t)n * (size_t)n;
    size_t batch = output->size;

    _Pragma("omp parallel for")
    for (size_t b = 0; b < batch; b++) {
        double* work = (double*)malloc(matrix_size * sizeof(double));
        if (!work) continue;

        size_t base = b * matrix_size;
        for (size_t i = 0; i < matrix_size; i++) {
            work[i] = get_value_as_double(input, base + i);
        }

        double det = 1.0;
        int sign = 1;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double pivot_abs = fabs(work[(size_t)col * n + col]);
            for (int row = col + 1; row < n; row++) {
                double candidate = fabs(work[(size_t)row * n + col]);
                if (candidate > pivot_abs) {
                    pivot_abs = candidate;
                    pivot = row;
                }
            }

            if (pivot_abs == 0.0) {
                det = 0.0;
                break;
            }

            if (pivot != col) {
                for (int j = 0; j < n; j++) {
                    double tmp = work[(size_t)col * n + j];
                    work[(size_t)col * n + j] = work[(size_t)pivot * n + j];
                    work[(size_t)pivot * n + j] = tmp;
                }
                sign = -sign;
            }

            double pivot_val = work[(size_t)col * n + col];
            det *= pivot_val;
            for (int row = col + 1; row < n; row++) {
                double factor = work[(size_t)row * n + col] / pivot_val;
                work[(size_t)row * n + col] = 0.0;
                for (int j = col + 1; j < n; j++) {
                    work[(size_t)row * n + j] -= factor * work[(size_t)col * n + j];
                }
            }
        }

        set_tensor_value_from_float(output, b, det * sign);
        free(work);
    }
}


// 实现 `unique` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
int unique_forward(const Tensor* input, Tensor* values, Tensor* indices, Tensor* inverse, Tensor* counts, int sorted) {
    if (!input || !values || !indices || !inverse || !counts) return 0;
    if (!input->data || !values->data || !indices->data || !inverse->data || !counts->data) return 0;
    if (values->size < input->size || indices->size < input->size || inverse->size < input->size || counts->size < input->size) return 0;

    size_t n = input->size;
    size_t elem_size = get_dtype_size(values->dtype);
    size_t* first_indices = (size_t*)malloc((n == 0 ? 1 : n) * sizeof(size_t));
    int* order = (int*)malloc((n == 0 ? 1 : n) * sizeof(int));
    int* remap = (int*)malloc((n == 0 ? 1 : n) * sizeof(int));
    if (!first_indices || !order || !remap) {
        free(first_indices);
        free(order);
        free(remap);
        return 0;
    }

    int unique_count = 0;
    for (size_t i = 0; i < n; i++) {
        int found = -1;
        for (int j = 0; j < unique_count; j++) {
            if (tensor_scalar_equal(input, i, first_indices[j])) {
                found = j;
                break;
            }
        }

        if (found < 0) {
            first_indices[unique_count] = i;
            copy_tensor_element(values, unique_count, input, i);
            set_tensor_value_from_int(indices, unique_count, (int64_t)i);
            set_tensor_value_from_int(counts, unique_count, 1);
            set_tensor_value_from_int(inverse, i, unique_count);
            unique_count++;
        } else {
            int64_t old_count = get_value_as_int64(counts, found);
            set_tensor_value_from_int(counts, found, old_count + 1);
            set_tensor_value_from_int(inverse, i, found);
        }
    }

    if (sorted && unique_count > 1) {
        for (int i = 0; i < unique_count; i++) order[i] = i;
        for (int i = 1; i < unique_count; i++) {
            int current = order[i];
            int j = i - 1;
            while (j >= 0 && tensor_scalar_compare(input, first_indices[order[j]], first_indices[current]) > 0) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = current;
        }

        void* tmp_values = malloc((size_t)unique_count * elem_size);
        int64_t* tmp_indices = (int64_t*)malloc((size_t)unique_count * sizeof(int64_t));
        int64_t* tmp_counts = (int64_t*)malloc((size_t)unique_count * sizeof(int64_t));
        if (tmp_values && tmp_indices && tmp_counts) {
            for (int new_pos = 0; new_pos < unique_count; new_pos++) {
                int old_pos = order[new_pos];
                remap[old_pos] = new_pos;
                memcpy((uint8_t*)tmp_values + (size_t)new_pos * elem_size,
                       (uint8_t*)values->data + (size_t)old_pos * elem_size,
                       elem_size);
                tmp_indices[new_pos] = get_value_as_int64(indices, old_pos);
                tmp_counts[new_pos] = get_value_as_int64(counts, old_pos);
            }

            memcpy(values->data, tmp_values, (size_t)unique_count * elem_size);
            for (int i = 0; i < unique_count; i++) {
                set_tensor_value_from_int(indices, i, tmp_indices[i]);
                set_tensor_value_from_int(counts, i, tmp_counts[i]);
            }
            for (size_t i = 0; i < n; i++) {
                int old_inverse = (int)get_value_as_int64(inverse, i);
                set_tensor_value_from_int(inverse, i, remap[old_inverse]);
            }
        }
        free(tmp_values);
        free(tmp_indices);
        free(tmp_counts);
    }

    free(first_indices);
    free(order);
    free(remap);
    return unique_count;
}


// 实现 `mel weight matrix` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void mel_weight_matrix_forward(const Tensor* num_mel_bins, const Tensor* dft_length,
                               const Tensor* sample_rate, const Tensor* lower_edge_hertz,
                               const Tensor* upper_edge_hertz, Tensor* output) {
    if (!num_mel_bins || !dft_length || !sample_rate || !lower_edge_hertz || !upper_edge_hertz || !output) return;

    int bins = (int)get_value_as_int64(num_mel_bins, 0);
    int dft_len = (int)get_value_as_int64(dft_length, 0);
    int rate = (int)get_value_as_int64(sample_rate, 0);
    double lower = get_value_as_double(lower_edge_hertz, 0);
    double upper = get_value_as_double(upper_edge_hertz, 0);
    if (bins < 0 || dft_len < 0 || rate <= 0 || upper < lower) return;

    int spectrogram_bins = dft_len / 2 + 1;
    if (output->ndim != 2 || output->shape[0] != spectrogram_bins || output->shape[1] != bins) return;

    double mel_lower = hz_to_mel(lower);
    double mel_upper = hz_to_mel(upper);

    for (int i = 0; i < bins; i++) {
        double left_mel = mel_lower + (mel_upper - mel_lower) * (double)i / (double)(bins + 1);
        double center_mel = mel_lower + (mel_upper - mel_lower) * (double)(i + 1) / (double)(bins + 1);
        double right_mel = mel_lower + (mel_upper - mel_lower) * (double)(i + 2) / (double)(bins + 1);

        int left = (int)floor((double)(dft_len + 1) * mel_to_hz(left_mel) / (double)rate);
        int center = (int)floor((double)(dft_len + 1) * mel_to_hz(center_mel) / (double)rate);
        int right = (int)floor((double)(dft_len + 1) * mel_to_hz(right_mel) / (double)rate);

        if (left < 0) left = 0;
        if (center < 0) center = 0;
        if (center > spectrogram_bins - 1) center = spectrogram_bins - 1;
        if (right < 0) right = 0;
        if (right > spectrogram_bins) right = spectrogram_bins;

        if (center == left && center >= 0 && center < spectrogram_bins) {
            set_tensor_value_from_float(output, (size_t)center * bins + i, 1.0);
        } else {
            for (int j = left; j <= center && j < spectrogram_bins; j++) {
                if (j >= 0) {
                    double value = (double)(j - left) / (double)(center - left);
                    set_tensor_value_from_float(output, (size_t)j * bins + i, value);
                }
            }
        }

        if (right > center) {
            for (int j = center; j < right && j < spectrogram_bins; j++) {
                if (j >= 0) {
                    double value = (double)(right - j) / (double)(right - center);
                    set_tensor_value_from_float(output, (size_t)j * bins + i, value);
                }
            }
        }
    }
}


// 实现 `dft` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dft_forward(const Tensor* input, Tensor* output, int axis, int inverse, int onesided, int dft_length) {
    if (!input || !output || !input->data || !output->data) return;
    if (input->ndim < 2 || output->ndim != input->ndim || input->ndim > MAX_NDIM) return;
    int complex_rank = input->ndim - 1;
    int input_complex_dim = input->shape[complex_rank];
    int output_complex_dim = output->shape[complex_rank];
    if (input_complex_dim != 1 && input_complex_dim != 2) return;
    if (output_complex_dim != 1 && output_complex_dim != 2) return;
    axis = normalize_complex_axis(axis, complex_rank);
    if (axis < 0 || axis >= complex_rank || dft_length <= 0) return;

    for (int d = 0; d < complex_rank; d++) {
        if (d != axis && input->shape[d] != output->shape[d]) return;
    }

    int input_axis_len = input->shape[axis];
    int output_axis_len = output->shape[axis];
    size_t vector_total = 1;
    for (int d = 0; d < complex_rank; d++) {
        if (d != axis) vector_total *= (size_t)output->shape[d];
    }

    _Pragma("omp parallel for collapse(2)")
    for (size_t vector_id = 0; vector_id < vector_total; vector_id++) {
        for (int k = 0; k < output_axis_len; k++) {
            int in_coords[MAX_NDIM] = {0};
            int out_coords[MAX_NDIM] = {0};
            size_t rem = vector_id;
            for (int d = complex_rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                int dim = output->shape[d];
                int coord = (int)(rem % (size_t)dim);
                rem /= (size_t)dim;
                in_coords[d] = coord;
                out_coords[d] = coord;
            }
            out_coords[axis] = k;

            if (inverse && onesided) {
                double real_sum = 0.0;
                int max_freq = dft_length / 2;
                for (int f = 0; f < input_axis_len; f++) {
                    if (f > max_freq) continue;
                    in_coords[axis] = f;
                    double xr, xi;
                    get_complex_value(input, in_coords, &xr, &xi);
                    double angle = TWO_PI * (double)f * (double)k / (double)dft_length;
                    double contribution = xr * cos(angle) - xi * sin(angle);
                    if (f != 0 && !(dft_length % 2 == 0 && f == dft_length / 2)) {
                        contribution *= 2.0;
                    }
                    real_sum += contribution;
                }
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum / (double)dft_length);
                continue;
            }

            double real_sum = 0.0;
            double imag_sum = 0.0;
            double sign = inverse ? 1.0 : -1.0;
            for (int n = 0; n < dft_length; n++) {
                double xr = 0.0;
                double xi = 0.0;
                if (n < input_axis_len) {
                    in_coords[axis] = n;
                    get_complex_value(input, in_coords, &xr, &xi);
                }
                double angle = sign * TWO_PI * (double)k * (double)n / (double)dft_length;
                double ca = cos(angle);
                double sa = sin(angle);
                real_sum += xr * ca - xi * sa;
                imag_sum += xr * sa + xi * ca;
            }
            if (inverse) {
                real_sum /= (double)dft_length;
                imag_sum /= (double)dft_length;
            }
            set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum);
            if (output_complex_dim == 2) {
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 1), imag_sum);
            }
        }
    }
}


// 实现 `stft` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void stft_forward(const Tensor* signal, const Tensor* window, Tensor* output,
                  int frame_step, int frame_length, int onesided) {
    if (!signal || !output || !signal->data || !output->data) return;
    if (signal->ndim < 2 || output->ndim != signal->ndim + 1 || signal->ndim + 1 > MAX_NDIM) return;
    if (frame_step <= 0 || frame_length <= 0) return;
    int signal_complex_rank = signal->ndim - 1;
    int output_complex_rank = output->ndim - 1;
    int prefix_rank = signal->ndim - 2;
    int signal_len = signal->shape[signal_complex_rank - 1];
    int signal_complex_dim = signal->shape[signal_complex_rank];
    int output_complex_dim = output->shape[output_complex_rank];
    if (signal_complex_dim != 1 && signal_complex_dim != 2) return;
    if (output_complex_dim != 2) return;
    for (int d = 0; d < prefix_rank; d++) {
        if (signal->shape[d] != output->shape[d]) return;
    }
    int n_frames = output->shape[prefix_rank];
    int bins = output->shape[prefix_rank + 1];
    int expected_bins = onesided ? frame_length / 2 + 1 : frame_length;
    if (bins != expected_bins) return;
    if (window && window->data && window->size < (size_t)frame_length) return;

    size_t prefix_total = 1;
    for (int d = 0; d < prefix_rank; d++) prefix_total *= (size_t)signal->shape[d];

    _Pragma("omp parallel for collapse(3)")
    for (size_t prefix_id = 0; prefix_id < prefix_total; prefix_id++) {
        for (int frame = 0; frame < n_frames; frame++) {
            for (int k = 0; k < bins; k++) {
                int sig_coords[MAX_NDIM] = {0};
                int out_coords[MAX_NDIM] = {0};
                size_t rem = prefix_id;
                for (int d = prefix_rank - 1; d >= 0; d--) {
                    int dim = signal->shape[d];
                    int coord = (int)(rem % (size_t)dim);
                    rem /= (size_t)dim;
                    sig_coords[d] = coord;
                    out_coords[d] = coord;
                }
                out_coords[prefix_rank] = frame;
                out_coords[prefix_rank + 1] = k;

                double real_sum = 0.0;
                double imag_sum = 0.0;
                for (int n = 0; n < frame_length; n++) {
                    int signal_pos = frame * frame_step + n;
                    double xr = 0.0;
                    double xi = 0.0;
                    if (signal_pos >= 0 && signal_pos < signal_len) {
                        sig_coords[prefix_rank] = signal_pos;
                        get_complex_value(signal, sig_coords, &xr, &xi);
                    }
                    double win = 1.0;
                    if (window && window->data) {
                        win = get_value_as_double(window, (size_t)n);
                    }
                    xr *= win;
                    xi *= win;
                    double angle = -TWO_PI * (double)k * (double)n / (double)frame_length;
                    double ca = cos(angle);
                    double sa = sin(angle);
                    real_sum += xr * ca - xi * sa;
                    imag_sum += xr * sa + xi * ca;
                }
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 0), real_sum);
                set_tensor_value_from_float(output, complex_tensor_index(output, out_coords, 1), imag_sum);
            }
        }
    }
}


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


// Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
// 实现 `hann window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hann_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return; // 甚至不需要写入
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = 0.5 * (1.0 - cos(2.0 * PI * i / denom));
        set_tensor_value_from_float(output, i, val);
    }
}


// Hamming Window: 0.54 - 0.46 * cos(2*pi*n / (N-1))
// 实现 `hamming window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void hamming_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0);
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double val = 0.54 - 0.46 * cos(2.0 * PI * i / denom);
        set_tensor_value_from_float(output, i, val);
    }
}


// Blackman Window: 0.42 - 0.5*cos(...) + 0.08*cos(...)
// 实现 `blackman window` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void blackman_window_forward(const Tensor* size_tensor, Tensor* output, int periodic) {
    if (!size_tensor || !output) return;
    int64_t N = get_window_size(size_tensor);
    if (N <= 0) return;
    if (N == 1) {
        set_tensor_value_from_float(output, 0, 1.0); // center value usually
        return;
    }

    double denom = periodic ? (double)N : (double)(N - 1);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)N; i++) {
        double term1 = 0.5 * cos(2.0 * PI * i / denom);
        double term2 = 0.08 * cos(4.0 * PI * i / denom);
        double val = 0.42 - term1 + term2;
        set_tensor_value_from_float(output, i, val);
    }
}

