/**
  ******************************************************************************
  * @file        verify_transpose.cu
  * @author      Egor Izmaylov
  * @brief       提供 transpose 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// 根据输出平坦索引反解输出坐标，再按 perm 映射回输入坐标并复制元素。
__global__ void transpose_kernel(
    const float* input,
    float* output,
    const int* input_shape,
    const int* perm,
    int rank,
    size_t out_len
) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= out_len || rank > MAX_RANK) return;

    int out_coords[MAX_RANK] = {0};
    int in_coords[MAX_RANK] = {0};

    size_t tmp = t;
    for (int d = rank - 1; d >= 0; --d) {
        int out_dim = input_shape[perm[d]];
        out_coords[d] = (int)(tmp % (size_t)out_dim);
        tmp /= (size_t)out_dim;
    }

    for (int d = 0; d < rank; ++d) {
        in_coords[perm[d]] = out_coords[d];
    }

    size_t in_idx = 0;
    for (int d = 0; d < rank; ++d) {
        in_idx = in_idx * (size_t)input_shape[d] + (size_t)in_coords[d];
    }

    output[t] = input[in_idx];
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];

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
    int perm[MAX_RANK] = {0};
    if (fread(input_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank ||
        fread(perm, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        fclose(fp);
        printf("read params failed\n");
        return 1;
    }
    fclose(fp);

    size_t in_len = 1;
    size_t expected_out_len = 1;
    for (int d = 0; d < rank; ++d) {
        if (input_shape[d] <= 0 || perm[d] < 0 || perm[d] >= rank) {
            printf("invalid shape or perm\n");
            return 1;
        }
        in_len *= (size_t)input_shape[d];
        expected_out_len *= (size_t)input_shape[perm[d]];
    }
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
    int* d_perm = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMalloc(&d_input_shape, (size_t)rank * sizeof(int));
    cudaMalloc(&d_perm, (size_t)rank * sizeof(int));

    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm, perm, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    transpose_kernel<<<blocks, threads>>>(d_input, d_output, d_input_shape, d_perm, rank, out_len);
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
    cudaFree(d_perm);
    free(h_input);
    free(h_output);
    return 0;
}
