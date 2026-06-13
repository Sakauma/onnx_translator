/**
  ******************************************************************************
  * @file        verify_unique.cu
  * @author      Egor Izmaylov
  * @brief       提供 Unique 算子的 CUDA 参考验证程序，输出 values/indices/inverse/counts。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct UniqueParams {
    int32_t type_code;
    int32_t sorted;
    int32_t input_len;
};

template <typename T>
__device__ int compare_value(T lhs, T rhs) {
    if (lhs < rhs) return -1;
    if (lhs > rhs) return 1;
    return 0;
}

template <typename T>
__global__ void unique_kernel(
    const T* input,
    T* values,
    int64_t* indices,
    int64_t* inverse,
    int64_t* counts,
    int* unique_count,
    T* tmp_values,
    int64_t* tmp_indices,
    int64_t* tmp_counts,
    int* order,
    int* remap,
    int n,
    int sorted
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int count = 0;
    for (int i = 0; i < n; ++i) {
        int found = -1;
        for (int j = 0; j < count; ++j) {
            if (compare_value(input[i], values[j]) == 0) {
                found = j;
                break;
            }
        }
        if (found < 0) {
            values[count] = input[i];
            indices[count] = i;
            counts[count] = 1;
            inverse[i] = count;
            ++count;
        } else {
            counts[found] += 1;
            inverse[i] = found;
        }
    }

    if (sorted && count > 1) {
        for (int i = 0; i < count; ++i) {
            order[i] = i;
        }
        for (int i = 1; i < count; ++i) {
            int current = order[i];
            int j = i - 1;
            while (j >= 0 && compare_value(values[order[j]], values[current]) > 0) {
                order[j + 1] = order[j];
                --j;
            }
            order[j + 1] = current;
        }

        for (int new_pos = 0; new_pos < count; ++new_pos) {
            int old_pos = order[new_pos];
            remap[old_pos] = new_pos;
            tmp_values[new_pos] = values[old_pos];
            tmp_indices[new_pos] = indices[old_pos];
            tmp_counts[new_pos] = counts[old_pos];
        }
        for (int i = 0; i < count; ++i) {
            values[i] = tmp_values[i];
            indices[i] = tmp_indices[i];
            counts[i] = tmp_counts[i];
        }
        for (int i = 0; i < n; ++i) {
            inverse[i] = remap[(int)inverse[i]];
        }
    }

    *unique_count = count;
}

static int read_params(const char* path, UniqueParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int ok = fread(params, sizeof(UniqueParams), 1, fp) == 1;
    fclose(fp);
    return ok;
}

template <typename T>
static int read_array(const char* path, T* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        return 0;
    }
    size_t got = fread(data, sizeof(T), n, fp);
    fclose(fp);
    return got == n;
}

template <typename T>
static int write_array(const char* path, const T* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        return 0;
    }
    size_t wrote = fwrite(data, sizeof(T), n, fp);
    fclose(fp);
    return wrote == n;
}

template <typename T>
static int run_unique(const char* input_path, const char* output_path, const UniqueParams& params) {
    int n = params.input_len;
    int expected_unique = 0;
    T* h_input = (T*)malloc((size_t)n * sizeof(T));
    T* h_values = (T*)malloc((size_t)n * sizeof(T));
    int64_t* h_indices = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    int64_t* h_inverse = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    int64_t* h_counts = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!h_input || !h_values || !h_indices || !h_inverse || !h_counts) return 0;
    if (!read_array(input_path, h_input, (size_t)n)) return 0;

    T* d_input = NULL;
    T* d_values = NULL;
    T* d_tmp_values = NULL;
    int64_t* d_indices = NULL;
    int64_t* d_inverse = NULL;
    int64_t* d_counts = NULL;
    int64_t* d_tmp_indices = NULL;
    int64_t* d_tmp_counts = NULL;
    int* d_count = NULL;
    int* d_order = NULL;
    int* d_remap = NULL;

    cudaMalloc(&d_input, (size_t)n * sizeof(T));
    cudaMalloc(&d_values, (size_t)n * sizeof(T));
    cudaMalloc(&d_tmp_values, (size_t)n * sizeof(T));
    cudaMalloc(&d_indices, (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_inverse, (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_counts, (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_tmp_indices, (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_tmp_counts, (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_order, (size_t)n * sizeof(int));
    cudaMalloc(&d_remap, (size_t)n * sizeof(int));

    cudaMemcpy(d_input, h_input, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    unique_kernel<<<1, 1>>>(
        d_input,
        d_values,
        d_indices,
        d_inverse,
        d_counts,
        d_count,
        d_tmp_values,
        d_tmp_indices,
        d_tmp_counts,
        d_order,
        d_remap,
        n,
        params.sorted
    );
    cudaMemcpy(&expected_unique, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (expected_unique < 0 || expected_unique > n) return 0;
    cudaMemcpy(h_values, d_values, (size_t)expected_unique * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, (size_t)expected_unique * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inverse, d_inverse, (size_t)n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, (size_t)expected_unique * sizeof(int64_t), cudaMemcpyDeviceToHost);

    int ok = 1;
    ok = ok && write_array(output_path, h_values, (size_t)expected_unique);
    ok = ok && write_array("tmp_unique_indices.bin", h_indices, (size_t)expected_unique);
    ok = ok && write_array("tmp_unique_inverse.bin", h_inverse, (size_t)n);
    ok = ok && write_array("tmp_unique_counts.bin", h_counts, (size_t)expected_unique);

    free(h_input);
    free(h_values);
    free(h_indices);
    free(h_inverse);
    free(h_counts);
    cudaFree(d_input);
    cudaFree(d_values);
    cudaFree(d_tmp_values);
    cudaFree(d_indices);
    cudaFree(d_inverse);
    cudaFree(d_counts);
    cudaFree(d_tmp_indices);
    cudaFree(d_tmp_counts);
    cudaFree(d_count);
    cudaFree(d_order);
    cudaFree(d_remap);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    UniqueParams params;
    if (!read_params(argv[3], &params)) return 1;
    if (params.input_len < 0) return 1;

    if (params.type_code == 1) {
        return run_unique<int64_t>(argv[2], argv[4], params) ? 0 : 1;
    }
    return run_unique<float>(argv[2], argv[4], params) ? 0 : 1;
}
