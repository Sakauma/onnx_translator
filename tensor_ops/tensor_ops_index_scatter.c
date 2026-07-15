/**
  ******************************************************************************
  * @file        tensor_ops_index_scatter.c
  * @author      Egor Izmaylov
  * @brief       实现 Gather、Scatter、NonZero 和索引写入类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// Gather 实现
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
        int out_coords[MAX_NDIM]; // 偷懒做法，最大维度不超过8
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

// ScatterND
// 遍历 updates，将其值写入 data 的指定位置
// 实现 `scatter nd` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void scatter_nd_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int reduction) {
    if (!data || !indices || !updates) return;
    
    int k = indices->shape[indices->ndim - 1]; 
    int r = data->ndim; 
    size_t loop_size = updates->size;
    int slice_ndim = r - k; 
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        int up_coords[MAX_NDIM];
        int data_coords[MAX_NDIM];
        int ind_coords[MAX_NDIM]; // indices 坐标
        
        // 反解 updates 坐标
        get_coords_from_index(i, up_coords, updates->shape, updates->ndim);
        
        // 构造 indices 的读取坐标
        for (int d = 0; d < indices->ndim - 1; d++) ind_coords[d] = up_coords[d];
        
        // 读取索引向量并构造 data 坐标前缀
        for (int j = 0; j < k; j++) {
            ind_coords[indices->ndim - 1] = j;
            size_t ind_idx = get_index_from_coords(ind_coords, indices->shape, indices->ndim);
            int64_t idx_val = get_value_as_int64(indices, ind_idx);
            
            // 处理负索引
            if (idx_val < 0) idx_val += data->shape[j];
            // 越界保护
            if (idx_val < 0) idx_val = 0;
            if (idx_val >= data->shape[j]) idx_val = data->shape[j] - 1;
            
            data_coords[j] = (int)idx_val;
        }
        
        // 补全 data 坐标后缀
        for (int j = 0; j < slice_ndim; j++) {
            data_coords[k + j] = up_coords[updates->ndim - slice_ndim + j];
        }
        
        // 计算目标索引
        size_t data_idx = get_index_from_coords(data_coords, data->shape, data->ndim);
        double val = get_value_as_double(updates, i);
        
        // 执行写入
        if (reduction == 0) {
            apply_scatter_update(data, data_idx, updates, i, reduction);
        } else if (reduction == 1) { // Add
            // 使用 switch-case 分发到具体类型以启用 omp atomic
            switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, +=)
                default: 
                    // 对于不支持 atomic 的类型，使用 critical
                    #pragma omp critical
                    {
                        apply_scatter_update(data, data_idx, updates, i, reduction);
                    }
                    break;
            }
        } else if (reduction == 2) { // Mul
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, *=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, *=)
                default:
                    #pragma omp critical
                    {
                        apply_scatter_update(data, data_idx, updates, i, reduction);
                    }
            }
        }
    }
}


// TensorScatter
// 复制 past_cache 后按 batch 级 write_indices 将 update 写入指定 sequence 轴，支持 linear/circular 模式。
void tensor_scatter_forward(const Tensor* past_cache, const Tensor* update, const Tensor* write_indices, Tensor* output, int axis, int mode) {
    if (!past_cache || !update || !output || !past_cache->data || !update->data || !output->data) return;
    if (past_cache->ndim != update->ndim || past_cache->ndim != output->ndim) return;
    if (past_cache->ndim <= 0 || past_cache->ndim > MAX_NDIM) return;

    int rank = past_cache->ndim;
    if (axis < 0) axis += rank;
    if (axis <= 0 || axis >= rank) return;

    for (int d = 0; d < rank; d++) {
        if (output->shape[d] != past_cache->shape[d]) return;
        if (d == axis) {
            if (update->shape[d] > past_cache->shape[d]) return;
        } else if (update->shape[d] != past_cache->shape[d]) {
            return;
        }
    }

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        copy_tensor_element(output, i, past_cache, i);
    }

    int max_sequence_length = past_cache->shape[axis];
    int sequence_length = update->shape[axis];
    if (max_sequence_length <= 0 || sequence_length < 0) return;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < update->size; i++) {
        int update_coords[MAX_NDIM] = {0};
        int target_coords[MAX_NDIM] = {0};
        get_coords_from_index(i, update_coords, update->shape, rank);
        for (int d = 0; d < rank; d++) {
            target_coords[d] = update_coords[d];
        }

        int batch_index = update_coords[0];
        int64_t write_start = write_indices ? get_value_as_int64(write_indices, (size_t)batch_index) : 0;
        int64_t target_sequence = write_start + update_coords[axis];
        if (mode == 1) {
            target_sequence %= max_sequence_length;
            if (target_sequence < 0) target_sequence += max_sequence_length;
        } else if (target_sequence < 0 || target_sequence >= max_sequence_length) {
            continue;
        }
        target_coords[axis] = (int)target_sequence;

        size_t target_idx = get_index_from_coords(target_coords, output->shape, rank);
        copy_tensor_element(output, target_idx, update, i);
    }
}


// GatherND
// 遍历 output，根据 indices 构造 data 坐标读取数据
// 实现 `gather nd` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gather_nd_forward(const Tensor* data, const Tensor* indices, Tensor* output, int batch_dims) {
    if (!data || !indices || !output) return;
    
    int k = indices->shape[indices->ndim - 1]; // 索引向量长度
    int r = data->ndim;
    int q = indices->ndim - 1; // indices 的前缀维度
    int slice_ndim = r - k - batch_dims; // 结果切片的维度数

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
        
        // indices 的坐标：前 batch_dims + (q - batch_dims) 来自 output
        for (int j = batch_dims; j < q; j++) {
            ind_coords[j] = out_coords[j];
        }
        
        // 读取 k 个索引值填充到 data_coords
        for (int j = 0; j < k; j++) {
            ind_coords[q] = j; // indices 最后一维
            size_t ind_idx = get_index_from_coords(ind_coords, indices->shape, indices->ndim);
            int64_t idx_val = get_value_as_int64(indices, ind_idx);
            
            // 维度偏移：data 的第 batch_dims + j 维
            int data_dim_idx = batch_dims + j;
            if (idx_val < 0) idx_val += data->shape[data_dim_idx];
            // 越界 clamp
            if (idx_val < 0) idx_val = 0;
            if (idx_val >= data->shape[data_dim_idx]) idx_val = data->shape[data_dim_idx] - 1;
            
            data_coords[data_dim_idx] = (int)idx_val;
        }
        
        // output 的最后 slice_ndim 维 对应 data 的最后 slice_ndim 维
        for (int j = 0; j < slice_ndim; j++) {
            data_coords[batch_dims + k + j] = out_coords[q + j];
        }
        
        size_t data_idx = get_index_from_coords(data_coords, data->shape, data->ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}


// GatherElements
// 实现 `gather elements` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void gather_elements_forward(const Tensor* data, const Tensor* indices, Tensor* output, int axis) {
    if (!data || !indices || !output) return;
    
    int ndim = data->ndim;
    if (axis < 0) axis += ndim;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 获取 index 值
        // indices 和 output 形状相同
        int64_t idx_val = get_value_as_int64(indices, i);
        if (idx_val < 0) idx_val += data->shape[axis];
        if (idx_val < 0) idx_val = 0;
        if (idx_val >= data->shape[axis]) idx_val = data->shape[axis] - 1;
        
        // 修改 axis 维度的坐标
        coords[axis] = (int)idx_val;
        
        size_t data_idx = get_index_from_coords(coords, data->shape, ndim);
        double val = get_value_as_double(data, data_idx);
        set_tensor_value_from_float(output, i, val);
    }
}

// ScatterElements
// 实现 `scatter elements` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void scatter_elements_forward(Tensor* data, const Tensor* indices, const Tensor* updates, int axis, int reduction) {
    if (!data || !indices || !updates) return;
    int ndim = data->ndim;
    if (axis < 0) axis += ndim;
    
    // 遍历 updates (和 indices 形状相同)
    size_t loop_size = updates->size;
    
    #pragma omp parallel for
    for (size_t i = 0; i < loop_size; i++) {
        int coords[MAX_NDIM];
        get_coords_from_index(i, coords, updates->shape, ndim);
        
        // 获取 index 值
        int64_t idx_val = get_value_as_int64(indices, i);
        if (idx_val < 0) idx_val += data->shape[axis];
        if (idx_val < 0) idx_val = 0;
        if (idx_val >= data->shape[axis]) idx_val = data->shape[axis] - 1;
        
        // 构造目标坐标: 除了 axis 维，其他与 updates 坐标一致
        coords[axis] = (int)idx_val;
        
        size_t data_idx = get_index_from_coords(coords, data->shape, ndim);
        double val = get_value_as_double(updates, i);
        
        if (reduction == 0) {
            apply_scatter_update(data, data_idx, updates, i, reduction);
        } else if (reduction == 1) { // Add
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, +=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, +=)
                default: 
                    #pragma omp critical
                    {
                        apply_scatter_update(data, data_idx, updates, i, reduction);
                    }
            }
        } else if (reduction == 2) { // Mul
             switch (data->dtype) {
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT32, float, *=)
                OMP_ATOMIC_DISPATCH(DTYPE_FLOAT64, double, *=)
                default:
                    #pragma omp critical
                    {
                        apply_scatter_update(data, data_idx, updates, i, reduction);
                    }
            }
        }
    }
}
