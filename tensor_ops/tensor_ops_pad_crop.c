/**
  ******************************************************************************
  * @file        tensor_ops_pad_crop.c
  * @author      Egor Izmaylov
  * @brief       实现 padding 和中心裁剪/填充类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从 shape/index shard 拆分 Pad 和 CenterCropPad。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// Pad
// mode: 0=constant, 1=reflect, 2=edge
// 实现 `pad` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void pad_forward(const Tensor* data, Tensor* output, const Tensor* pads, const Tensor* constant_value, int mode) {
    if (!data || !output || !pads) return;

    int ndim = data->ndim;

    int64_t pad_begins[MAX_NDIM];
    for (int d = 0; d < ndim; d++) {
        pad_begins[d] = get_value_as_int64(pads, d);
    }

    double const_val = 0.0;
    if (constant_value && constant_value->data) {
        const_val = get_value_as_double(constant_value, 0);
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0};
        int in_coords[MAX_NDIM] = {0};
        int in_bounds = 1; // 标记是否在源数据范围内

        get_coords_from_index(i, out_coords, output->shape, ndim);

        for (int d = 0; d < ndim; d++) {
            // 计算相对于源数据的坐标
            int64_t c = out_coords[d] - pad_begins[d];
            int64_t dim_len = data->shape[d];

            if (c >= 0 && c < dim_len) {
                // 在范围内
                in_coords[d] = (int)c;
            } else {
                // 在 Padding 区域
                if (mode == 0) { // Constant
                    in_bounds = 0;
                    break;
                } else if (mode == 2) { // Edge
                    if (c < 0) c = 0;
                    if (c >= dim_len) c = dim_len - 1;
                    in_coords[d] = (int)c;
                } else if (mode == 1) { // Reflect
                    if (dim_len <= 1) {
                        c = 0;
                    } else {
                        int64_t M = 2 * dim_len - 2;
                        int64_t k = c % M;
                        if (k < 0) k += M;
                        if (k >= dim_len) {
                            k = M - k;
                        }
                        c = k;
                    }
                    in_coords[d] = (int)c;
                } else if (mode == 3) { // Wrap
                    if (dim_len <= 0) {
                        in_bounds = 0;
                        break;
                    }
                    c %= dim_len;
                    if (c < 0) c += dim_len;
                    in_coords[d] = (int)c;
                }
            }
        }

        if (in_bounds) {
            size_t in_idx = get_index_from_coords(in_coords, data->shape, ndim);
            copy_tensor_element(output, i, data, in_idx);
        } else {
            if (constant_value && constant_value->data) {
                copy_tensor_element(output, i, constant_value, 0);
            } else {
                set_tensor_value_from_float(output, i, const_val);
            }
        }
    }
}


// CenterCropPad
// 根据输入和输出 shape 的差值执行官方中心裁剪/零填充，奇数 padding 额外像素落在右侧。
void center_crop_pad_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data) return;
    if (input->ndim != output->ndim || input->ndim > MAX_NDIM) return;

    int rank = input->ndim;
    int crop_starts[MAX_NDIM] = {0};
    int pad_begins[MAX_NDIM] = {0};

    for (int d = 0; d < rank; d++) {
        int input_dim = input->shape[d];
        int output_dim = output->shape[d];
        if (input_dim < 0 || output_dim < 0) return;
        if (input_dim > output_dim) {
            crop_starts[d] = (input_dim - output_dim) / 2;
            pad_begins[d] = 0;
        } else {
            crop_starts[d] = 0;
            pad_begins[d] = (output_dim - input_dim) / 2;
        }
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0};
        int in_coords[MAX_NDIM] = {0};
        int in_bounds = 1;

        get_coords_from_index(i, out_coords, output->shape, rank);
        for (int d = 0; d < rank; d++) {
            int src_coord = out_coords[d] - pad_begins[d] + crop_starts[d];
            if (src_coord < 0 || src_coord >= input->shape[d]) {
                in_bounds = 0;
                break;
            }
            in_coords[d] = src_coord;
        }

        if (in_bounds) {
            size_t in_idx = get_index_from_coords(in_coords, input->shape, rank);
            copy_tensor_element(output, i, input, in_idx);
        } else {
            set_tensor_value_from_float(output, i, 0.0);
        }
    }
}
