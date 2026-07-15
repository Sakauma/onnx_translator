/**
  ******************************************************************************
  * @file        tensor_ops_gather.c
  * @author      Egor Izmaylov
  * @brief       实现 Gather 算子族的 C 后端入口。
  * @details     2026.07.15  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 实现 `gather` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gather_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;

    int ndim_data = data->ndim;
    int ndim_indices = indices->ndim;
    int ndim_out = output->ndim;

    if (axis < 0) axis += ndim_data;
    if (axis < 0 || axis >= ndim_data) return;

    int axis_dim_limit = data->shape[axis];

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int data_coords[MAX_NDIM];
        int indices_coords[MAX_NDIM];

        get_coords_from_index(i, out_coords, output->shape, ndim_out);
        for (int j = 0; j < ndim_indices; j++) {
            indices_coords[j] = out_coords[axis + j];
        }

        size_t idx_idx = get_index_from_coords(indices_coords, indices->shape, ndim_indices);
        int64_t index_val = get_value_as_int64(indices, idx_idx);

        if (index_val < 0) index_val += axis_dim_limit;
        if (index_val < 0 || index_val >= axis_dim_limit) index_val = 0;

        for (int j = 0; j < axis; j++) {
            data_coords[j] = out_coords[j];
        }
        data_coords[axis] = (int)index_val;
        for (int j = axis + 1; j < ndim_data; j++) {
            data_coords[j] = out_coords[j - 1 + ndim_indices];
        }

        size_t data_idx = get_index_from_coords(data_coords, data->shape, ndim_data);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// 实现 `gather nd` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gather_nd_forward(const Tensor* data, const Tensor* indices, Tensor* output, int batch_dims) {
    if (!data || !indices || !output) return;

    int k = indices->shape[indices->ndim - 1];
    int r = data->ndim;
    int q = indices->ndim - 1;
    int slice_ndim = r - k - batch_dims;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int ind_coords[MAX_NDIM];
        int data_coords[MAX_NDIM];

        get_coords_from_index(i, out_coords, output->shape, output->ndim);
        for (int b = 0; b < batch_dims; b++) {
            data_coords[b] = out_coords[b];
            ind_coords[b] = out_coords[b];
        }

        for (int j = batch_dims; j < q; j++) {
            ind_coords[j] = out_coords[j];
        }

        for (int j = 0; j < k; j++) {
            ind_coords[q] = j;
            size_t ind_idx = get_index_from_coords(ind_coords, indices->shape, indices->ndim);
            int64_t idx_val = get_value_as_int64(indices, ind_idx);

            int data_dim_idx = batch_dims + j;
            if (idx_val < 0) idx_val += data->shape[data_dim_idx];
            if (idx_val < 0) idx_val = 0;
            if (idx_val >= data->shape[data_dim_idx]) idx_val = data->shape[data_dim_idx] - 1;

            data_coords[data_dim_idx] = (int)idx_val;
        }

        for (int j = 0; j < slice_ndim; j++) {
            data_coords[batch_dims + k + j] = out_coords[q + j];
        }

        size_t data_idx = get_index_from_coords(data_coords, data->shape, data->ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// 实现 `gather elements` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gather_elements_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;

    int ndim = data->ndim;
    if (axis < 0) axis += ndim;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);

        int64_t idx_val = get_value_as_int64(indices, i);
        if (idx_val < 0) idx_val += data->shape[axis];
        if (idx_val < 0) idx_val = 0;
        if (idx_val >= data->shape[axis]) idx_val = data->shape[axis] - 1;

        coords[axis] = (int)idx_val;

        size_t data_idx = get_index_from_coords(coords, data->shape, ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}
