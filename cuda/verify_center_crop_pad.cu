/**
  ******************************************************************************
  * @file        verify_center_crop_pad.cu
  * @author      Egor Izmaylov
  * @brief       提供 CenterCropPad 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// 根据输出坐标反推居中裁剪/零填充后的输入坐标，越界位置写入 0。
__global__ void center_crop_pad_kernel(
    const float* input,
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
        int crop_start = input_shape[d] > output_shape[d] ? (input_shape[d] - output_shape[d]) / 2 : 0;
        int pad_begin = input_shape[d] < output_shape[d] ? (output_shape[d] - input_shape[d]) / 2 : 0;
        int src_coord = out_coords[d] - pad_begin + crop_start;
        if (src_coord < 0 || src_coord >= input_shape[d]) {
            in_bounds = 0;
            break;
        }
        in_coords[d] = src_coord;
    }

    if (!in_bounds) {
        output[t] = 0.0f;
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

    int rank = 0;
    if (fread(&rank, sizeof(int), 1, fp) != 1 || rank <= 0 || rank > MAX_RANK) {
        fclose(fp);
        printf("invalid rank\n");
        return 1;
    }

    int input_shape[MAX_RANK] = {0};
    int output_shape[MAX_RANK] = {0};
    if (fread(input_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank ||
        fread(output_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        fclose(fp);
        printf("read params failed\n");
        return 1;
    }
    fclose(fp);

    for (int d = 0; d < rank; ++d) {
        if (input_shape[d] <= 0 || output_shape[d] < 0) {
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
    float* h_output = (float*)malloc(out_bytes == 0 ? sizeof(float) : out_bytes);
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

    if (out_len > 0) {
        float* d_input = NULL;
        float* d_output = NULL;
        int* d_input_shape = NULL;
        int* d_output_shape = NULL;
        cudaMalloc(&d_input, in_bytes);
        cudaMalloc(&d_output, out_bytes);
        cudaMalloc(&d_input_shape, (size_t)rank * sizeof(int));
        cudaMalloc(&d_output_shape, (size_t)rank * sizeof(int));

        cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_shape, input_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_shape, output_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
        center_crop_pad_kernel<<<blocks, threads>>>(d_input, d_output, d_input_shape, d_output_shape, rank, out_len);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_input_shape);
        cudaFree(d_output_shape);
    }

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

    free(h_input);
    free(h_output);
    return 0;
}
