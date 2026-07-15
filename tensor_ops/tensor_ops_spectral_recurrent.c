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

#include "internal/sequence.h"


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
