/**
  ******************************************************************************
  * @file        tensor_ops_sort_scan.c
  * @author      Egor Izmaylov
  * @brief       实现 TopK、累计扫描和 Einsum 类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 降序比较函数
// 作为 `compare_desc` 排序比较函数，保证排序类算子的值和索引顺序稳定。
int compare_desc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;

    int nan1 = isnan(e1->value);
    int nan2 = isnan(e2->value);
    
    if (nan1 && nan2) return (e1->index < e2->index) ? -1 : 1;
    if (nan1) return -1; 
    if (nan2) return 1; 

    if (e1->value > e2->value) return -1;
    if (e1->value < e2->value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 升序比较函数
// 作为 `compare_asc` 排序比较函数，保证排序类算子的值和索引顺序稳定。
int compare_asc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;

    int nan1 = isnan(e1->value);
    int nan2 = isnan(e2->value);
    
    if (nan1 && nan2) return (e1->index < e2->index) ? -1 : 1;
    if (nan1) return 1; 
    if (nan2) return -1;

    if (e1->value < e2->value) return -1;
    if (e1->value > e2->value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 作为 `compare_signed_desc` 排序比较函数，按有符号整数值降序排列，平局时保留较小原始索引。
int compare_signed_desc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;
    if (e1->signed_value > e2->signed_value) return -1;
    if (e1->signed_value < e2->signed_value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 作为 `compare_signed_asc` 排序比较函数，按有符号整数值升序排列，平局时保留较小原始索引。
int compare_signed_asc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;
    if (e1->signed_value < e2->signed_value) return -1;
    if (e1->signed_value > e2->signed_value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 作为 `compare_unsigned_desc` 排序比较函数，按无符号整数值降序排列，平局时保留较小原始索引。
int compare_unsigned_desc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;
    if (e1->raw_value > e2->raw_value) return -1;
    if (e1->raw_value < e2->raw_value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 作为 `compare_unsigned_asc` 排序比较函数，按无符号整数值升序排列，平局时保留较小原始索引。
int compare_unsigned_asc(const void* a, const void* b) {
    TopKElement* e1 = (TopKElement*)a;
    TopKElement* e2 = (TopKElement*)b;
    if (e1->raw_value < e2->raw_value) return -1;
    if (e1->raw_value > e2->raw_value) return 1;
    return (e1->index < e2->index) ? -1 : 1;
}


// 实现 `topk` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void topk_forward(const Tensor* input, Tensor* values, Tensor* indices, int axis, int largest, int sorted, int K) {
    if (!input || !values || !indices) return;
    (void)sorted;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    int axis_dim = input->shape[axis];
    int outer_loops = 1;
    for (int i = 0; i < axis; i++) outer_loops *= input->shape[i];
    int inner_loops = 1;
    for (int i = axis + 1; i < ndim; i++) inner_loops *= input->shape[i];
    int integer_path = is_integer_dtype(input->dtype) && is_integer_dtype(values->dtype);
    int unsigned_path = is_unsigned_integer_dtype(input->dtype);
    
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)outer_loops * inner_loops; i++) {
        // 计算当前处理的 row 的位置
        int inner_idx = i % inner_loops;
        int outer_idx = i / inner_loops;
        
        // 临时 buffer，存放该轴的所有元素
        TopKElement* buffer = (TopKElement*)malloc(axis_dim * sizeof(TopKElement));
        if (!buffer) continue;
        
        // 读取数据
        for (int k = 0; k < axis_dim; k++) {
            // 构造完整坐标的 flat index
            // Index = outer * (axis_dim * inner) + k * inner + inner_idx
            size_t idx = (size_t)outer_idx * axis_dim * inner_loops + (size_t)k * inner_loops + inner_idx;
            if (integer_path) {
                buffer[k].raw_value = get_integer_value_as_uint64(input, idx);
                buffer[k].signed_value = get_value_as_int64(input, idx);
            } else {
                buffer[k].value = get_value_as_double(input, idx);
            }
            buffer[k].index = k; // 记录原始下标
        }
        
        // 排序
        if (integer_path && unsigned_path) {
            qsort(buffer, axis_dim, sizeof(TopKElement), largest ? compare_unsigned_desc : compare_unsigned_asc);
        } else if (integer_path) {
            qsort(buffer, axis_dim, sizeof(TopKElement), largest ? compare_signed_desc : compare_signed_asc);
        } else if (largest) {
            qsort(buffer, axis_dim, sizeof(TopKElement), compare_desc);
        } else {
            qsort(buffer, axis_dim, sizeof(TopKElement), compare_asc);
        }
        
        // 写入前 K 个
        int write_k = (K < axis_dim) ? K : axis_dim;
        for (int k = 0; k < write_k; k++) {
            // Output shape is same as Input except axis=K
            // OutIndex = outer * (K * inner) + k * inner + inner_idx
            size_t out_idx = (size_t)outer_idx * K * inner_loops + (size_t)k * inner_loops + inner_idx;
            
            if (integer_path) {
                set_integer_value_wrapped(values, out_idx, buffer[k].raw_value);
            } else {
                set_tensor_value_from_float(values, out_idx, buffer[k].value);
            }
            set_tensor_value_from_int(indices, out_idx, buffer[k].index);
        }
        free(buffer);
    }
}


// 实现 `cumsum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void cumsum_forward(const Tensor* input, Tensor* output, int axis, int exclusive, int reverse) {
    if (!input || !output) return;
    
    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    
    int axis_dim = input->shape[axis];
    int outer_loops = 1;
    for (int i = 0; i < axis; i++) outer_loops *= input->shape[i];
    int inner_loops = 1;
    for (int i = axis + 1; i < ndim; i++) inner_loops *= input->shape[i];
    int integer_path = is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype);
    
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)outer_loops * inner_loops; i++) {
        int inner_idx = i % inner_loops;
        int outer_idx = i / inner_loops;
        
        double accumulator = 0.0;
        uint64_t integer_accumulator = 0;
        
        // 确定遍历方向
        int start = reverse ? axis_dim - 1 : 0;
        int end   = reverse ? -1 : axis_dim;
        int step  = reverse ? -1 : 1;
        
        for (int k = start; k != end; k += step) {
            size_t idx = (size_t)outer_idx * axis_dim * inner_loops + (size_t)k * inner_loops + inner_idx;
            if (integer_path) {
                uint64_t val = get_integer_value_as_uint64(input, idx);
                if (exclusive) {
                    set_integer_value_wrapped(output, idx, integer_accumulator);
                    integer_accumulator += val;
                } else {
                    integer_accumulator += val;
                    set_integer_value_wrapped(output, idx, integer_accumulator);
                }
            } else {
                double val = get_value_as_double(input, idx);
                if (exclusive) {
                    set_tensor_value_from_float(output, idx, accumulator);
                    accumulator += val;
                } else {
                    accumulator += val;
                    set_tensor_value_from_float(output, idx, accumulator);
                }
            }
        }
    }
}


// 实现 `cumprod` 算子的 C 后端入口，按指定轴计算累计乘积并按目标 dtype 写回。
void cumprod_forward(const Tensor* input, Tensor* output, int axis, int exclusive, int reverse) {
    if (!input || !output) return;

    int ndim = input->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return;

    int axis_dim = input->shape[axis];
    int outer_loops = 1;
    for (int i = 0; i < axis; i++) outer_loops *= input->shape[i];
    int inner_loops = 1;
    for (int i = axis + 1; i < ndim; i++) inner_loops *= input->shape[i];
    int integer_path = is_integer_dtype(input->dtype) && is_integer_dtype(output->dtype);

    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)outer_loops * inner_loops; i++) {
        int inner_idx = i % inner_loops;
        int outer_idx = i / inner_loops;

        double accumulator = 1.0;
        uint64_t integer_accumulator = 1;

        int start = reverse ? axis_dim - 1 : 0;
        int end = reverse ? -1 : axis_dim;
        int step = reverse ? -1 : 1;

        for (int k = start; k != end; k += step) {
            size_t idx = (size_t)outer_idx * axis_dim * inner_loops + (size_t)k * inner_loops + inner_idx;
            if (integer_path) {
                uint64_t val = get_integer_value_as_uint64(input, idx);
                if (exclusive) {
                    set_integer_value_wrapped(output, idx, integer_accumulator);
                    integer_accumulator *= val;
                } else {
                    integer_accumulator *= val;
                    set_integer_value_wrapped(output, idx, integer_accumulator);
                }
            } else {
                double val = get_value_as_double(input, idx);
                if (exclusive) {
                    set_tensor_value_from_float(output, idx, accumulator);
                    accumulator *= val;
                } else {
                    accumulator *= val;
                    set_tensor_value_from_float(output, idx, accumulator);
                }
            }
        }
    }
}


// 实现 `einsum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void einsum_forward(const Tensor** inputs, int num_inputs, Tensor* output, 
                    int iter_dims, int* loop_limits, 
                    int* input_strides, int* output_strides) {
    
    // 总迭代次数
    size_t total_ops = 1;
    for (int i = 0; i < iter_dims; i++) total_ops *= loop_limits[i];
    size_t out_size = output->size;
    
    double* accum = (double*)calloc(out_size, sizeof(double));
    if (!accum) return;
    
    // 并行化大循环
    #pragma omp parallel for
    for (size_t op = 0; op < total_ops; op++) {
        // 反解当前的循环计数器 (counters)
        // counters[d] 代表第 d 个“标签”当前的索引值
        // 假设 iter_dims 不会超过 26 (a-z)
        int counters[26]; 
        size_t temp_op = op;
        for (int d = iter_dims - 1; d >= 0; d--) {
            counters[d] = temp_op % loop_limits[d];
            temp_op /= loop_limits[d];
        }
        
        // 计算每个输入的 Flat Index
        // Index_k = Sum_d ( counters[d] * stride_k[d] )
        double product = 1.0;
        
        for (int k = 0; k < num_inputs; k++) {
            size_t in_idx = 0;
            int* cur_strides = &input_strides[k * iter_dims];
            
            for (int d = 0; d < iter_dims; d++) {
                in_idx += counters[d] * cur_strides[d];
            }
            
            product *= get_value_as_double(inputs[k], in_idx);
        }
        
        // 计算输出的 Flat Index
        size_t out_idx = 0;
        for (int d = 0; d < iter_dims; d++) {
            out_idx += counters[d] * output_strides[d];
        }
        
        #pragma omp atomic
        accum[out_idx] += product;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < out_size; i++) {
        set_tensor_value_from_float(output, i, accum[i]);
    }

    free(accum);
}
