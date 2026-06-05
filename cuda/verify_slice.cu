/**
  ******************************************************************************
  * @file        verify_slice.cu
  * @author      Egor Izmaylov
  * @brief       提供 Slice 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 保存 Slice 参考计算所需的输入/输出形状，以及归一化后的 starts 和 steps。
struct SliceParams {
    int32_t rank;
    int32_t input_shape[MAX_RANK];
    int32_t output_shape[MAX_RANK];
    int32_t starts[MAX_RANK];
    int32_t steps[MAX_RANK];
};

// 按输出坐标映射回输入坐标，验证 C 后端 slice_forward 的坐标搬运逻辑。
__global__ void slice_kernel(const float* input, float* output, SliceParams p, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    size_t tmp = tid;
    size_t input_index = 0;
    size_t input_stride = 1;

    for (int d = p.rank - 1; d >= 0; --d) {
        int32_t out_dim = p.output_shape[d];
        int32_t out_coord = out_dim > 0 ? (int32_t)(tmp % (size_t)out_dim) : 0;
        if (out_dim > 0) {
            tmp /= (size_t)out_dim;
        }
        int32_t input_coord = p.starts[d] + out_coord * p.steps[d];
        input_index += (size_t)input_coord * input_stride;
        input_stride *= (size_t)p.input_shape[d];
    }

    output[tid] = input[input_index];
}

// 顺序读取 `[rank, input_shape..., output_shape..., starts..., steps...]` 参数。
static int read_slice_params(const char* params_path, SliceParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    if (fread(&params->rank, sizeof(int32_t), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "read rank failed\n");
        return 0;
    }
    if (params->rank <= 0 || params->rank > MAX_RANK) {
        fclose(fp);
        fprintf(stderr, "invalid rank\n");
        return 0;
    }
    int32_t* groups[] = {
        params->input_shape,
        params->output_shape,
        params->starts,
        params->steps,
    };
    for (int group = 0; group < 4; ++group) {
        if (fread(groups[group], sizeof(int32_t), (size_t)params->rank, fp) != (size_t)params->rank) {
            fclose(fp);
            fprintf(stderr, "read params group failed\n");
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <data.bin> <starts.bin> <ends.bin> <axes.bin> <steps.bin> <params.bin> <out.bin>
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <starts.bin> <ends.bin> <axes.bin> <steps.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* params_path = argv[7];
    const char* out_path = argv[8];

    SliceParams params;
    if (!read_slice_params(params_path, &params)) {
        return 1;
    }

    size_t input_len = 1;
    size_t expected_out_len = 1;
    for (int d = 0; d < params.rank; ++d) {
        input_len *= (size_t)params.input_shape[d];
        expected_out_len *= (size_t)params.output_shape[d];
    }
    if (out_len != expected_out_len) {
        fprintf(stderr, "out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    float* h_input = (float*)malloc(input_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* input_fp = fopen(data_path, "rb");
    if (!input_fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t read_count = fread(h_input, sizeof(float), input_len, input_fp);
    fclose(input_fp);
    if (read_count != input_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, input_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMemcpy(d_input, h_input, input_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    slice_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t write_count = fwrite(h_output, sizeof(float), out_len, out_fp);
    fclose(out_fp);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
