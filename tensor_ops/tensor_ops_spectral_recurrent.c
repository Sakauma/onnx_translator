/**
  ******************************************************************************
  * @file        tensor_ops_spectral_recurrent.c
  * @author      Egor Izmaylov
  * @brief       实现谱分析、矩阵统计和窗口函数类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
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
    double mel_step = (mel_upper - mel_lower) / (double)(bins + 2);

    for (int i = 0; i < bins; i++) {
        double left_mel = mel_lower + mel_step * (double)i;
        double center_mel = mel_lower + mel_step * (double)(i + 1);
        double right_mel = mel_lower + mel_step * (double)(i + 2);

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


// Hamming Window: alpha - beta * cos(2*pi*n / (N-1)), alpha=25/46, beta=21/46
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
        double val = (25.0 / 46.0) - (21.0 / 46.0) * cos(2.0 * PI * i / denom);
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
