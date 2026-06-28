/**
  ******************************************************************************
  * @file        tensor_ops_spectral_transform.c
  * @author      Egor Izmaylov
  * @brief       实现 DFT 和 STFT 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从谱分析 shard 拆分频域变换实现。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


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
