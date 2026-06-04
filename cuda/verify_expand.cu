/**
  ******************************************************************************
  * @file        verify_expand.cu
  * @author      Egor Izmaylov
  * @brief       提供 expand 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// 根据输出坐标和左侧补 1 后的输入形状，将广播维度映射回输入坐标。
__global__ void expand_kernel(
    const float* input,
    float* output,
    const int* input_shape,
    const int* output_shape,
    int rank_in,
    int rank_out,
    size_t out_len
) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= out_len || rank_in > MAX_RANK || rank_out > MAX_RANK) return;

    int out_coords[MAX_RANK] = {0};
    int in_coords[MAX_RANK] = {0};
    int offset = rank_out - rank_in;

    size_t tmp = t;
    for (int d = rank_out - 1; d >= 0; --d) {
        out_coords[d] = (int)(tmp % (size_t)output_shape[d]);
        tmp /= (size_t)output_shape[d];
    }

    for (int d = 0; d < rank_in; ++d) {
        int out_axis = d + offset;
        in_coords[d] = (input_shape[d] == 1) ? 0 : out_coords[out_axis];
    }

    size_t in_idx = 0;
    for (int d = 0; d < rank_in; ++d) {
        in_idx = in_idx * (size_t)input_shape[d] + (size_t)in_coords[d];
    }
    output[t] = input[in_idx];
}

// 计算形状数组对应的元素数量。
static size_t shape_size(const int* shape, int rank) {
    size_t size = 1;
    for (int d = 0; d < rank; ++d) {
        size *= (size_t)shape[d];
    }
    return size;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <shape.bin> <params.bin> <out.bin>
    if (argc != 6) {
        printf("Usage: %s <out_len> <input.bin> <shape.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        printf("open params failed\n");
        return 1;
    }

    int ranks[2] = {0};
    if (fread(ranks, sizeof(int), 2, fp) != 2) {
        fclose(fp);
        printf("read ranks failed\n");
        return 1;
    }
    int rank_in = ranks[0];
    int rank_out = ranks[1];
    if (rank_in <= 0 || rank_in > MAX_RANK || rank_out <= 0 || rank_out > MAX_RANK || rank_out < rank_in) {
        fclose(fp);
        printf("invalid ranks\n");
        return 1;
    }

    int input_shape[MAX_RANK] = {0};
    int output_shape[MAX_RANK] = {0};
    if (fread(input_shape, sizeof(int), (size_t)rank_in, fp) != (size_t)rank_in ||
        fread(output_shape, sizeof(int), (size_t)rank_out, fp) != (size_t)rank_out) {
        fclose(fp);
        printf("read shapes failed\n");
        return 1;
    }
    fclose(fp);

    for (int d = 0; d < rank_in; ++d) {
        if (input_shape[d] <= 0) {
            printf("invalid input shape\n");
            return 1;
        }
    }
    for (int d = 0; d < rank_out; ++d) {
        if (output_shape[d] <= 0) {
            printf("invalid output shape\n");
            return 1;
        }
    }

    size_t in_len = shape_size(input_shape, rank_in);
    size_t expected_out_len = shape_size(output_shape, rank_out);
    if (out_len != expected_out_len) {
        printf("out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    size_t in_bytes = in_len * sizeof(float);
    size_t out_bytes = out_len * sizeof(float);
    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    if (!h_input || !h_output) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        printf("open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), in_len, fi);
    fclose(fi);
    if (ri != in_len) {
        printf("fread mismatch\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    int* d_input_shape = NULL;
    int* d_output_shape = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMalloc(&d_input_shape, (size_t)rank_in * sizeof(int));
    cudaMalloc(&d_output_shape, (size_t)rank_out * sizeof(int));

    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape, (size_t)rank_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shape, output_shape, (size_t)rank_out * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    expand_kernel<<<blocks, threads>>>(d_input, d_output, d_input_shape, d_output_shape, rank_in, rank_out, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        printf("open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) {
        printf("fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_shape);
    cudaFree(d_output_shape);
    free(h_input);
    free(h_output);
    return 0;
}
