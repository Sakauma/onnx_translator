/**
  ******************************************************************************
  * @file        verify_pad.cu
  * @author      Egor Izmaylov
  * @brief       提供 pad 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// 根据输出坐标反推输入坐标，越界位置按 constant mode 写入常量。
__global__ void pad_constant_kernel(
    const float* input,
    const long long* pads,
    const float* constant_value,
    float* output,
    const int* input_shape,
    const int* output_shape,
    int rank,
    size_t out_len
) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= out_len || rank > MAX_RANK) return;

    int out_coords[MAX_RANK] = {0};
    int in_coords[MAX_RANK] = {0};
    size_t tmp = t;
    for (int d = rank - 1; d >= 0; --d) {
        out_coords[d] = (int)(tmp % (size_t)output_shape[d]);
        tmp /= (size_t)output_shape[d];
    }

    int in_bounds = 1;
    for (int d = 0; d < rank; ++d) {
        long long c = (long long)out_coords[d] - pads[d];
        if (c < 0 || c >= (long long)input_shape[d]) {
            in_bounds = 0;
            break;
        }
        in_coords[d] = (int)c;
    }

    if (!in_bounds) {
        output[t] = constant_value[0];
        return;
    }

    size_t in_idx = 0;
    for (int d = 0; d < rank; ++d) {
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
    // <out_len> <data.bin> <pads.bin> <constant.bin> <params.bin> <out.bin>
    if (argc != 7) {
        printf("Usage: %s <out_len> <data.bin> <pads.bin> <constant.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* pads_path = argv[3];
    const char* constant_path = argv[4];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        printf("open params failed\n");
        return 1;
    }

    int header[2] = {0};
    if (fread(header, sizeof(int), 2, fp) != 2) {
        fclose(fp);
        printf("read header failed\n");
        return 1;
    }
    int rank = header[0];
    int mode = header[1];
    if (rank <= 0 || rank > MAX_RANK || mode != 0) {
        fclose(fp);
        printf("invalid rank or unsupported mode\n");
        return 1;
    }

    int input_shape[MAX_RANK] = {0};
    int output_shape[MAX_RANK] = {0};
    if (fread(input_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank ||
        fread(output_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        fclose(fp);
        printf("read shapes failed\n");
        return 1;
    }
    fclose(fp);

    for (int d = 0; d < rank; ++d) {
        if (input_shape[d] <= 0 || output_shape[d] <= 0) {
            printf("invalid shape\n");
            return 1;
        }
    }

    size_t in_len = shape_size(input_shape, rank);
    size_t expected_out_len = shape_size(output_shape, rank);
    if (out_len != expected_out_len) {
        printf("out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    size_t in_bytes = in_len * sizeof(float);
    size_t out_bytes = out_len * sizeof(float);
    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    long long* h_pads = (long long*)malloc((size_t)rank * 2 * sizeof(long long));
    float h_constant[1] = {0.0f};
    if (!h_input || !h_output || !h_pads) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(data_path, "rb");
    FILE* fpads = fopen(pads_path, "rb");
    FILE* fc = fopen(constant_path, "rb");
    if (!fi || !fpads || !fc) {
        printf("open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), in_len, fi);
    size_t rp = fread(h_pads, sizeof(long long), (size_t)rank * 2, fpads);
    size_t rc = fread(h_constant, sizeof(float), 1, fc);
    fclose(fi);
    fclose(fpads);
    fclose(fc);
    if (ri != in_len || rp != (size_t)rank * 2 || rc != 1) {
        printf("fread mismatch\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    float* d_constant = NULL;
    long long* d_pads = NULL;
    int* d_input_shape = NULL;
    int* d_output_shape = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMalloc(&d_constant, sizeof(float));
    cudaMalloc(&d_pads, (size_t)rank * 2 * sizeof(long long));
    cudaMalloc(&d_input_shape, (size_t)rank * sizeof(int));
    cudaMalloc(&d_output_shape, (size_t)rank * sizeof(int));

    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant, h_constant, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, h_pads, (size_t)rank * 2 * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shape, output_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    pad_constant_kernel<<<blocks, threads>>>(
        d_input,
        d_pads,
        d_constant,
        d_output,
        d_input_shape,
        d_output_shape,
        rank,
        out_len
    );
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
    cudaFree(d_constant);
    cudaFree(d_pads);
    cudaFree(d_input_shape);
    cudaFree(d_output_shape);
    free(h_input);
    free(h_output);
    free(h_pads);
    return 0;
}
