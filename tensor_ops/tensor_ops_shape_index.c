/**
  ******************************************************************************
  * @file        tensor_ops_shape_index.c
  * @author      Egor Izmaylov
  * @brief       实现形状变换和尺寸调整类 C 后端算子。
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


// 实现 `size` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void size_forward(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    int64_t total_elems = (int64_t)input->size;
    set_tensor_value_from_int(output, 0, total_elems);
}
