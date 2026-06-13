/**
  ******************************************************************************
  * @file        verify_split.cu
  * @author      Egor Izmaylov
  * @brief       提供 Split 算子的 CUDA 参考验证程序，支持多输出拼接校验。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 8
#define MAX_OUTPUTS 16

struct SplitParams {
    int32_t rank;
    int32_t axis;
    int32_t num_outputs;
    int32_t dims[MAX_RANK];
    int32_t split_sizes[MAX_OUTPUTS];
    int64_t output_offsets[MAX_OUTPUTS];
};

__global__ void split_kernel(const float* input, float* output, SplitParams params, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    int out_id = 0;
    for (int i = 0; i < params.num_outputs; ++i) {
        int64_t start = params.output_offsets[i];
        int64_t end = start;
        int64_t piece = params.split_sizes[i];
        for (int r = 0; r < params.rank; ++r) {
            if (r != params.axis) {
                piece *= params.dims[r];
            }
        }
        end += piece;
        if ((int64_t)tid >= start && (int64_t)tid < end) {
            out_id = i;
            break;
        }
    }

    int64_t local = (int64_t)tid - params.output_offsets[out_id];
    int local_coords[MAX_RANK] = {0};
    for (int r = params.rank - 1; r >= 0; --r) {
        int dim = (r == params.axis) ? params.split_sizes[out_id] : params.dims[r];
        local_coords[r] = (int)(local % dim);
        local /= dim;
    }

    int axis_offset = 0;
    for (int i = 0; i < out_id; ++i) {
        axis_offset += params.split_sizes[i];
    }
    local_coords[params.axis] += axis_offset;

    int64_t input_index = 0;
    for (int r = 0; r < params.rank; ++r) {
        input_index = input_index * params.dims[r] + local_coords[r];
    }
    output[tid] = input[input_index];
}

static int read_params(const char* path, SplitParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t header[3];
    if (fread(header, sizeof(int32_t), 3, fp) != 3) {
        fclose(fp);
        return 0;
    }
    params->rank = header[0];
    params->axis = header[1];
    params->num_outputs = header[2];
    if (params->rank <= 0 || params->rank > MAX_RANK || params->num_outputs <= 0 || params->num_outputs > MAX_OUTPUTS) {
        fclose(fp);
        return 0;
    }
    if (fread(params->dims, sizeof(int32_t), params->rank, fp) != (size_t)params->rank) {
        fclose(fp);
        return 0;
    }
    if (fread(params->split_sizes, sizeof(int32_t), params->num_outputs, fp) != (size_t)params->num_outputs) {
        fclose(fp);
        return 0;
    }
    fclose(fp);

    int64_t offset = 0;
    for (int i = 0; i < params->num_outputs; ++i) {
        params->output_offsets[i] = offset;
        int64_t piece = params->split_sizes[i];
        for (int r = 0; r < params->rank; ++r) {
            if (r != params->axis) {
                piece *= params->dims[r];
            }
        }
        offset += piece;
    }
    return 1;
}

static int read_f32_file(const char* path, float* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        return 0;
    }
    size_t got = fread(data, sizeof(float), n, fp);
    fclose(fp);
    return got == n;
}

static int write_f32_file(const char* path, const float* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        return 0;
    }
    size_t wrote = fwrite(data, sizeof(float), n, fp);
    fclose(fp);
    return wrote == n;
}

int main(int argc, char** argv) {
    if (argc < 6) return 1;

    size_t out_len = (size_t)atoll(argv[1]);
    SplitParams params;
    if (!read_params(argv[4], &params)) return 1;

    int64_t input_len = 1;
    for (int r = 0; r < params.rank; ++r) {
        input_len *= params.dims[r];
    }

    float* h_x = (float*)malloc((size_t)input_len * sizeof(float));
    float* h_out = (float*)malloc(out_len * sizeof(float));
    if (!h_x || !h_out) return 1;
    if (!read_f32_file(argv[2], h_x, (size_t)input_len)) return 1;

    float* d_x = NULL;
    float* d_out = NULL;
    cudaMalloc(&d_x, (size_t)input_len * sizeof(float));
    cudaMalloc(&d_out, out_len * sizeof(float));
    cudaMemcpy(d_x, h_x, (size_t)input_len * sizeof(float), cudaMemcpyHostToDevice);

    split_kernel<<<(out_len + 255) / 256, 256>>>(d_x, d_out, params, out_len);
    cudaMemcpy(h_out, d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    int ok = write_f32_file(argv[5], h_out, out_len);

    free(h_x);
    free(h_out);
    cudaFree(d_x);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
