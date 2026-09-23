/**
  * @file        tensor_ops_einsum.c
  * @brief       Implements the Einsum C backend.
  */

#include "tensor_ops_internal.h"

// 实现 `einsum` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void einsum_forward(const Tensor** inputs, int num_inputs, Tensor* output,
                    int iter_dims, int* loop_limits,
                    int* input_strides, int* output_strides) {

    // 总迭代次数
    size_t total_ops = 1;
    for (int i = 0; i < iter_dims; i++) total_ops *= loop_limits[i];
    size_t out_size = output->size;

    // Einsum 的整数输入/输出按目标 dtype 回绕。用无符号乘加避免 signed
    // overflow 的未定义行为，也避免 int64/uint64 经 double 丢失低位。
    int integer_path = is_integer_dtype(output->dtype);
    for (int k = 0; k < num_inputs && integer_path; k++) {
        integer_path = inputs[k] && is_integer_dtype(inputs[k]->dtype);
    }
    if (integer_path) {
        uint64_t* integer_accum = (uint64_t*)calloc(out_size, sizeof(uint64_t));
        if (!integer_accum) return;

        #pragma omp parallel for
        for (size_t op = 0; op < total_ops; op++) {
            int counters[26];
            size_t temp_op = op;
            for (int d = iter_dims - 1; d >= 0; d--) {
                counters[d] = temp_op % loop_limits[d];
                temp_op /= loop_limits[d];
            }

            uint64_t product = 1ULL;
            for (int k = 0; k < num_inputs; k++) {
                size_t in_idx = 0;
                int* cur_strides = &input_strides[k * iter_dims];
                for (int d = 0; d < iter_dims; d++) {
                    in_idx += counters[d] * cur_strides[d];
                }
                product *= get_integer_value_as_uint64(inputs[k], in_idx);
            }

            size_t out_idx = 0;
            for (int d = 0; d < iter_dims; d++) {
                out_idx += counters[d] * output_strides[d];
            }
            #pragma omp atomic
            integer_accum[out_idx] += product;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < out_size; i++) {
            set_integer_value_wrapped(output, i, integer_accum[i]);
        }
        free(integer_accum);
        return;
    }

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
