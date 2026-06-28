/**
  ******************************************************************************
  * @file        tensor_ops_shape_index.c
  * @author      Egor Izmaylov
  * @brief       实现形状变换和布局转换类 C 后端算子。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// Flatten 实现
// 实现 `flatten` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void flatten_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || input->size != output->size) return;
    size_t elem_size = get_dtype_size(input->dtype);
    size_t total_bytes = input->size * elem_size;
    memcpy(output->data, input->data, total_bytes);
}


// Reshape 实现
// 实现 `reshape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void reshape_forward(const Tensor* input, Tensor* output) {
    flatten_forward(input, output);
}


// Transpose 实现
// 实现 `transpose` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void transpose_forward(const Tensor* input, Tensor* output, int* perm) {
    if (!input || !output || !perm) return;
    int ndim = input->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0}; // 输出坐标
        int in_coords[MAX_NDIM] = {0};  // 输入坐标
        
        // 1. 根据输出的平坦索引 i，反解出输出坐标
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        // 2. 映射回输入坐标
        // 规则：output[d] 对应 input[perm[d]]
        for (int k = 0; k < ndim; k++) {
            in_coords[perm[k]] = out_coords[k];
        }
        
        // 3. 计算输入的平坦索引
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        
        // 4. 搬运数据
        double val = get_value_as_double(input, in_idx);
        set_tensor_value_from_float(output, i, val);
    }
}


// 实现 `concat` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void concat_forward(const Tensor** inputs, int num_inputs, Tensor* output, int axis) {
    if (!inputs || !output || num_inputs < 1) return;

    // 处理负轴
    int ndim = output->ndim;
    if (ndim > MAX_NDIM) {

        return;
    }
    
    // 缓存每个输入在 axis 维度的长度
    int input_dims[128]; // 假设输入数量不超过 128
    if (num_inputs > 128) return; 
    for (int k = 0; k < num_inputs; k++) {
        input_dims[k] = inputs[k]->shape[axis];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int coords[MAX_NDIM]; // 最大维度为 16
        
        // 1. 反解输出坐标
        get_coords_from_index(i, coords, output->shape, ndim);
        
        // 2. 确定当前坐标落在哪个输入张量中
        int target_val = coords[axis];
        int input_idx = -1;
        int local_axis_val = target_val;
        
        for (int k = 0; k < num_inputs; k++) {
            if (local_axis_val < input_dims[k]) {
                input_idx = k;
                break;
            }
            local_axis_val -= input_dims[k];
        }
        
        if (input_idx >= 0) {
            // 3. 修正为局部坐标
            coords[axis] = local_axis_val;
            
            // 4. 读取源数据并写入
            const Tensor* src = inputs[input_idx];
            size_t src_idx = get_index_from_coords(coords, src->shape, ndim);
            copy_tensor_element(output, i, src, src_idx);
        }
    }
}


// 实现 `slice` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void slice_forward(const Tensor* input, Tensor* output, int* starts, int* steps) {
    if (!input || !output || !starts || !steps) return;
    
    int ndim = input->ndim;
    if (ndim > MAX_NDIM) {
        return;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
        
        // 1. 获取输出坐标
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        // 2. 映射回输入坐标: in = start + out * step
        for (int d = 0; d < ndim; d++) {
            in_coords[d] = starts[d] + out_coords[d] * steps[d];
        }
        
        // 3. 读写数据
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        copy_tensor_element(output, i, input, in_idx);
    }
}


// Cast
// 读取时自动转 double，写入 set_tensor_value 时会自动转为 output->dtype
// 实现 `cast` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void cast_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) return;

    if (input->dtype == output->dtype) {
        size_t elem_size = get_dtype_size(input->dtype);
        memcpy(output->data, input->data, input->size * elem_size);
        return;
    }
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < input->size; i++) {
        set_tensor_value_for_cast(output, i, input, i);
    }
}


// BitCast 不做数值转换，只在输入输出 dtype 等宽时复制底层字节。
void bitcast_forward(const Tensor* input, Tensor* output) {
    if (!input || !output || !input->data || !output->data || input->size != output->size) return;

    size_t in_elem_size = get_dtype_size(input->dtype);
    size_t out_elem_size = get_dtype_size(output->dtype);
    if (in_elem_size != out_elem_size) return;

    memcpy(output->data, input->data, input->size * in_elem_size);
}


// 实现 `eye like` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void eye_like_forward(Tensor* output, int k) {
    if (!output || output->ndim != 2) return;
    int cols = output->shape[1];

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int row = (int)(i / (size_t)cols);
        int col = (int)(i % (size_t)cols);
        double value = (col == row + k) ? 1.0 : 0.0;
        set_tensor_value_from_float(output, i, value);
    }
}


// Expand 实现
// 实现 `expand` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void expand_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim_in = input->ndim;
    int ndim_out = output->ndim;
    
    // 维度差 
    int offset = ndim_out - ndim_in;

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        int in_coords[MAX_NDIM];
        
        get_coords_from_index(i, out_coords, output->shape, ndim_out);
        
        // 映射回输入坐标
        for (int d = 0; d < ndim_in; d++) {
            int out_dim_idx = d + offset; // 对应输出的维度索引
            // 如果输入在该维度是1，则坐标固定为0（广播）；否则随输出变化
            if (input->shape[d] == 1) {
                in_coords[d] = 0;
            } else {
                in_coords[d] = out_coords[out_dim_idx];
            }
        }
        
        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim_in);
        copy_tensor_element(output, i, input, in_idx);
    }
}


// Shape 实现
// 实现 `shape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void shape_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    // Output 应该是一个 1D int64 张量，长度等于 input->ndim
    int64_t* out_data = (int64_t*)output->data;
    for (int i = 0; i < input->ndim; i++) {
        out_data[i] = (int64_t)input->shape[i];
    }
}


// ConstantOfShape
// 实现 `constant of shape` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void constant_of_shape_forward(Tensor* output, const Tensor* value) {
    if (!output) return;

    size_t loop_size = output->size;
    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        if (value && value->data) {
            copy_tensor_element(output, i, value, 0);
        } else {
            set_tensor_value_from_float(output, i, 0.0);
        }
    }
}


// Range
// 实现 `range` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void range_forward(const Tensor* start, const Tensor* limit, const Tensor* delta, Tensor* output) {
    if (!start || !limit || !delta || !output) return;

    size_t loop_size = output->size;
    if (IS_INT_TYPE(output->dtype) && IS_INT_TYPE(start->dtype) && IS_INT_TYPE(delta->dtype)) {
        uint64_t val_start = get_integer_value_as_uint64(start, 0);
        uint64_t val_delta = get_integer_value_as_uint64(delta, 0);
        _Pragma("omp parallel for")
        for (size_t i = 0; i < loop_size; i++) {
            uint64_t res = val_start + (uint64_t)i * val_delta;
            set_integer_value_wrapped(output, i, res);
        }
        return;
    }

    double val_start = get_value_as_double(start, 0);
    double val_delta = get_value_as_double(delta, 0);

    _Pragma("omp parallel for")
    for (size_t i = 0; i < loop_size; i++) {
        double res = val_start + (double)i * val_delta;
        set_tensor_value_from_float(output, i, res);
    }
}


// Tile
// 输入坐标 = 输出坐标 % 输入维度
// 实现 `tile` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void tile_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    
    int ndim = input->ndim;

    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM] = {0};
        int in_coords[MAX_NDIM] = {0};
        
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        for (int d = 0; d < ndim; d++) {
            in_coords[d] = out_coords[d] % input->shape[d];
        }

        size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
        copy_tensor_element(output, i, input, in_idx);
    }
}


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


// Resize
// 实现 `resize` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void resize_forward(const Tensor* input, Tensor* output, float* scales, int coord_mode, int mode, int nearest_mode) {
    if (!input || !output || !scales) return;
    
    int ndim = input->ndim;
    
    _Pragma("omp parallel for")
    for (size_t i = 0; i < output->size; i++) {
        int out_coords[MAX_NDIM];
        get_coords_from_index(i, out_coords, output->shape, ndim);
        
        if (mode == 0) { 
            // --- Nearest Neighbor ---
            int in_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;
                
                // 坐标变换
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f; // half_pixel
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f; // pytorch_half_pixel
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f; // align_corners
                else x_in = x_out / scale; // asymmetric (default)
                
                // 最近邻取整策略
                int in_idx = 0;
                if (nearest_mode == 2) { 
                    // floor
                    in_idx = (int)floorf(x_in);
                } else if (nearest_mode == 3) { 
                    // ceil
                    in_idx = (int)ceilf(x_in);
                } else {
                    // round_prefer_floor
                    in_idx = (int)ceilf(x_in - 0.5f);
                }
                // 边界截断 (Clamp)
                if (in_idx < 0) in_idx = 0;
                if (in_idx >= input->shape[d]) in_idx = input->shape[d] - 1;
                in_coords[d] = in_idx;
            }
            size_t in_idx = get_index_from_coords(in_coords, input->shape, ndim);
            double val = get_value_as_double(input, in_idx);
            set_tensor_value_from_float(output, i, val);
            
        } else {
            // --- Linear Interpolation (N-Linear) ---
            // 计算每个维度的浮点坐标 x_in
            float real_coords[MAX_NDIM];
            for (int d = 0; d < ndim; d++) {
                float x_out = (float)out_coords[d];
                float scale = scales[d];
                float x_in = 0.0f;
                if (coord_mode == 0) x_in = (x_out + 0.5f) / scale - 0.5f;
                else if (coord_mode == 2) x_in = (output->shape[d] > 1) ? (x_out + 0.5f) / scale - 0.5f : 0.0f;
                else if (coord_mode == 4) x_in = (output->shape[d] > 1) ? x_out * (input->shape[d] - 1) / (float)(output->shape[d] - 1) : 0.0f;
                else x_in = x_out / scale;
                
                if (x_in < 0.0f) x_in = 0.0f;
                if (x_in > (float)(input->shape[d] - 1)) x_in = (float)(input->shape[d] - 1);
                
                real_coords[d] = x_in;
            }
            // N-Linear 插值核心
            int num_neighbors = 1 << ndim; // 2^ndim
            double weighted_sum = 0.0;
            for (int n = 0; n < num_neighbors; n++) {
                double weight = 1.0;
                int neighbor_coords[MAX_NDIM];
                for (int d = 0; d < ndim; d++) {
                    float x = real_coords[d];
                    int lower = (int)floorf(x);
                    int upper = lower + 1;
                    if (upper >= input->shape[d]) upper = input->shape[d] - 1; 
                    // 检查当前邻居在维度 d 是取 Lower 还是 Upper
                    if ((n >> d) & 1) {
                        // 取 Upper
                        neighbor_coords[d] = upper;
                        weight *= (x - lower); 
                    } else {
                        // 取 Lower
                        neighbor_coords[d] = lower;
                        weight *= (1.0f - (x - lower)); 
                    }
                }
                size_t n_idx = get_index_from_coords(neighbor_coords, input->shape, ndim);
                double val = get_value_as_double(input, n_idx);
                weighted_sum += val * weight;
            }
            set_tensor_value_from_float(output, i, weighted_sum);
        }
    }
}



// 实现 `size` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void size_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int64_t total_elems = (int64_t)input->size;
    set_tensor_value_from_int(output, 0, total_elems);
}


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
