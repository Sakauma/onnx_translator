/**
  ******************************************************************************
  * @file        verify_compress.cu
  * @author      Egor Izmaylov
  * @brief       提供 Compress 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 8

// 保存 Compress 参考计算所需的输入/输出 rank、shape、axis 和 condition 长度。
struct CompressParams {
    int32_t input_rank;
    int32_t output_rank;
    int32_t axis;
    int32_t condition_len;
    int32_t input_shape[MAX_RANK];
    int32_t output_shape[MAX_RANK];
};

// axis 为 -1 时按官方 flatten 模式执行压缩。
__global__ void compress_flat_kernel(const float* input, const int32_t* index_map, float* output, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;
    output[tid] = input[index_map[tid]];
}

// axis 模式下按输出坐标反推输入坐标。
__global__ void compress_axis_kernel(
    const float* input,
    const int32_t* index_map,
    float* output,
    CompressParams p,
    size_t out_len
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    size_t tmp = tid;
    size_t input_index = 0;
    size_t input_stride = 1;

    for (int d = p.input_rank - 1; d >= 0; --d) {
        int32_t out_dim = p.output_shape[d];
        int32_t coord = out_dim > 0 ? (int32_t)(tmp % (size_t)out_dim) : 0;
        if (out_dim > 0) {
            tmp /= (size_t)out_dim;
        }
        int32_t input_coord = d == p.axis ? index_map[coord] : coord;
        input_index += (size_t)input_coord * input_stride;
        input_stride *= (size_t)p.input_shape[d];
    }

    output[tid] = input[input_index];
}

// 顺序读取 `[input_rank, output_rank, axis, condition_len, input_shape..., output_shape...]` 参数。
static int read_compress_params(const char* params_path, CompressParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t header[4];
    if (fread(header, sizeof(int32_t), 4, fp) != 4) {
        fclose(fp);
        fprintf(stderr, "read header failed\n");
        return 0;
    }
    params->input_rank = header[0];
    params->output_rank = header[1];
    params->axis = header[2];
    params->condition_len = header[3];
    if (params->input_rank <= 0 || params->input_rank > MAX_RANK || params->output_rank <= 0 || params->output_rank > MAX_RANK) {
        fclose(fp);
        fprintf(stderr, "invalid rank\n");
        return 0;
    }
    if (fread(params->input_shape, sizeof(int32_t), (size_t)params->input_rank, fp) != (size_t)params->input_rank) {
        fclose(fp);
        fprintf(stderr, "read input shape failed\n");
        return 0;
    }
    if (fread(params->output_shape, sizeof(int32_t), (size_t)params->output_rank, fp) != (size_t)params->output_rank) {
        fclose(fp);
        fprintf(stderr, "read output shape failed\n");
        return 0;
    }
    fclose(fp);
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <data.bin> <condition.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <condition.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* condition_path = argv[3];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    CompressParams params;
    if (!read_compress_params(params_path, &params)) {
        return 1;
    }

    size_t input_len = 1;
    size_t expected_out_len = 1;
    for (int d = 0; d < params.input_rank; ++d) {
        input_len *= (size_t)params.input_shape[d];
    }
    for (int d = 0; d < params.output_rank; ++d) {
        expected_out_len *= (size_t)params.output_shape[d];
    }
    if (out_len != expected_out_len) {
        fprintf(stderr, "out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    float* h_input = (float*)malloc(input_len * sizeof(float));
    float* h_condition = (float*)malloc((size_t)params.condition_len * sizeof(float));
    int32_t* h_index_map = (int32_t*)malloc((size_t)params.condition_len * sizeof(int32_t));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_input || !h_condition || !h_index_map || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* data_fp = fopen(data_path, "rb");
    FILE* cond_fp = fopen(condition_path, "rb");
    if (!data_fp || !cond_fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t data_read = fread(h_input, sizeof(float), input_len, data_fp);
    size_t cond_read = fread(h_condition, sizeof(float), (size_t)params.condition_len, cond_fp);
    fclose(data_fp);
    fclose(cond_fp);
    if (data_read != input_len || cond_read != (size_t)params.condition_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    int32_t kept = 0;
    for (int32_t i = 0; i < params.condition_len; ++i) {
        if (h_condition[i] != 0.0f) {
            h_index_map[kept++] = i;
        }
    }
    if ((size_t)kept < out_len && params.axis == -1) {
        fprintf(stderr, "condition keeps fewer elements than output\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    int32_t* d_index_map = NULL;
    cudaMalloc(&d_input, input_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMalloc(&d_index_map, (size_t)kept * sizeof(int32_t));
    cudaMemcpy(d_input, h_input, input_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_map, h_index_map, (size_t)kept * sizeof(int32_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    if (params.axis == -1) {
        compress_flat_kernel<<<blocks, threads>>>(d_input, d_index_map, d_output, out_len);
    } else {
        compress_axis_kernel<<<blocks, threads>>>(d_input, d_index_map, d_output, params, out_len);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t written = fwrite(h_output, sizeof(float), out_len, out_fp);
    fclose(out_fp);
    if (written != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_index_map);
    free(h_input);
    free(h_condition);
    free(h_index_map);
    free(h_output);
    return 0;
}
