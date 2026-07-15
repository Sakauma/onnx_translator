/**
  ******************************************************************************
  * @file        tensor_ops_layout_sequence.c
  * @author      Egor Izmaylov
  * @brief       实现布局转换、矩阵三角和序列选择类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "internal/spatial.h"


// OneHot
// indices: 输入索引
// values: [off_value, on_value] (2 element tensor)
// axis: 扩充的维度
// 实现 `one hot` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void one_hot_forward(const Tensor* indices, const Tensor* values, Tensor* output, int axis) {
    if (!indices || !values || !output) return;
    
    int out_ndim = output->ndim;
    if (axis < 0) axis += out_ndim;
    
    int depth = output->shape[axis];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int idx_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, out_ndim);
        
        int k = 0;
        for (int d = 0; d < out_ndim; d++) {
            if (d != axis) {
                idx_coords[k++] = out_coords[d];
            }
        }
        size_t idx_idx = get_index_from_coords(idx_coords, indices->shape, indices->ndim);
        int64_t target_idx = get_value_as_int64(indices, idx_idx);
        
        if (target_idx < 0) target_idx += depth;
        
        int current_depth_idx = out_coords[axis];
        
        copy_tensor_element(output, i, values, (current_depth_idx == target_idx) ? 1 : 0);
    }
}


// Tril / Triu
// 实现 `triangular` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void triangular_forward(const Tensor* input, Tensor* output, int k, int upper) {
    if (!input || !output) return;
    int ndim = input->ndim;
    if (ndim < 2) return; 
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, input->shape, ndim);
        
        int row = coords[ndim - 2];
        int col = coords[ndim - 1];
        
        double val = get_value_as_double(input, i);
        double res = 0.0;
        
        if (upper) {
            if (col - row >= k) res = val;
            else res = 0.0;
        } else {
            if (col - row <= k) res = val;
            else res = 0.0;
        }
        set_tensor_value_from_float(output, i, res);
    }
}


// DepthToSpace
// 实现 `depth to space` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void depth_to_space_forward(const Tensor* input, Tensor* output, int blocksize, int mode) {
    if (!input || !output) return;
    
    int N = input->shape[0];
    int C_out = output->shape[1];
    int H_out = output->shape[2];
    int W_out = output->shape[3];
    
    // 遍历输出坐标
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    // 反推输入坐标
                    // 输出坐标 (h, w) 对应 spatial block 中的 (dy, dx)
                    int in_h = h / blocksize;
                    int dy = h % blocksize;
                    int in_w = w / blocksize;
                    int dx = w % blocksize;
                    
                    int in_c = 0;
                    if (mode == 0) { // DCR: depth = [dy, dx, c]
                        // C dimension composed of (blocksize, blocksize, C_out)
                        in_c = (dy * blocksize + dx) * C_out + c;
                    } else { // CRD: depth = [c, dy, dx]
                        // C dimension composed of (C_out, blocksize, blocksize)
                        in_c = c * (blocksize * blocksize) + (dy * blocksize + dx);
                    }
                    
                    double val = get_val_4d_with_padding(input, n, in_c, in_h, in_w, 0.0);
                    
                    size_t out_idx = ((size_t)n * C_out * H_out * W_out) + 
                                     ((size_t)c * H_out * W_out) + 
                                     ((size_t)h * W_out) + w;
                    set_tensor_value_from_float(output, out_idx, val);
                }
            }
        }
    }
}


// SpaceToDepth
// 实现 `space to depth` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void space_to_depth_forward(const Tensor* input, Tensor* output, int blocksize) {
    if (!input || !output) return;
    
    int N = output->shape[0];
    int C_out = output->shape[1];
    int H_out = output->shape[2];
    int W_out = output->shape[3];
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            int C_in = input->shape[1];
            int in_c = c % C_in;
            int rem = c / C_in;
            int dy = rem / blocksize;
            int dx = rem % blocksize;
            
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int in_h = h * blocksize + dy;
                    int in_w = w * blocksize + dx;
                    
                    double val = get_val_4d_with_padding(input, n, in_c, in_h, in_w, 0.0);
                    
                    size_t out_idx = ((size_t)n * C_out * H_out * W_out) + 
                                     ((size_t)c * H_out * W_out) + 
                                     ((size_t)h * W_out) + w;
                    set_tensor_value_from_float(output, out_idx, val);
                }
            }
        }
    }
}


// ReverseSequence
// 实现 `reverse sequence` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void reverse_sequence_forward(const Tensor* input, const Tensor* sequence_lens, Tensor* output, int time_axis, int batch_axis) {
    if (!input || !output || !sequence_lens) return;
    int ndim = input->ndim;
    if (time_axis < 0) time_axis += ndim;
    if (batch_axis < 0) batch_axis += ndim;
    
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->size * elem_size);
    
    size_t strides[MAX_NDIM];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) strides[i] = strides[i+1] * input->shape[i+1];
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        int b_idx = coords[batch_axis];
        int t_idx = coords[time_axis];
        
        int64_t seq_len = get_value_as_int64(sequence_lens, b_idx);
        
        if (t_idx < seq_len) {
            int old_t_idx = (int)seq_len - 1 - t_idx;
            coords[time_axis] = old_t_idx;
            
            size_t src_idx = get_index_from_coords(coords, input->shape, ndim);
            double val = get_value_as_double(input, src_idx);
            set_tensor_value_from_float(output, i, val);
        }
    }
}


// Compress
// 实现 `compress` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void compress_forward(const Tensor* input, const Tensor* condition, Tensor* output, int axis) {
    if (!input || !condition || !output) return;
    int ndim = input->ndim;
    int flatten_mode = axis < -ndim;
    if (!flatten_mode && axis < 0) axis += ndim;
    if (!flatten_mode && (axis < 0 || axis >= ndim)) return;
    
    int cond_len = condition->size;
    int* idx_map = (int*)malloc(cond_len * sizeof(int));
    if (!idx_map) return;
    int count = 0;
    for (int i = 0; i < cond_len; i++) {
        if (get_value_as_double(condition, i) != 0.0) {
            idx_map[count++] = i;
        }
    }

    if (flatten_mode) {
        #pragma omp parallel for
        for (size_t i = 0; i < output->size; i++) {
            int src_idx = idx_map[i];
            if (src_idx >= 0 && (size_t)src_idx < input->size) {
                copy_tensor_element(output, i, input, (size_t)src_idx);
            }
        }
        free(idx_map);
        return;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM] = {0};
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 映射 axis 坐标
        int out_axis_idx = coords[axis];
        if (out_axis_idx < count) {
            coords[axis] = idx_map[out_axis_idx]; // 替换为原坐标
            
            size_t src_idx = get_index_from_coords(coords, input->shape, ndim);
            copy_tensor_element(output, i, input, src_idx);
        }
    }
    
    free(idx_map);
}
