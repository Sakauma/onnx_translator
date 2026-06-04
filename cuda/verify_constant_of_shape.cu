/**
  ******************************************************************************
  * @file        verify_constant_of_shape.cu
  * @author      Egor Izmaylov
  * @brief       提供 constant_of_shape 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// ConstantOfShape 对输出中每个元素写入同一个标量常量。
__global__ void constant_of_shape_kernel(float* output, float fill_value, size_t out_len) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < out_len) {
        output[t] = fill_value;
    }
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
    // <out_len> <shape.bin> <params.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <shape.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        printf("open params failed\n");
        return 1;
    }

    int rank = 0;
    if (fread(&rank, sizeof(int), 1, fp) != 1 || rank < 0 || rank > MAX_RANK) {
        fclose(fp);
        printf("invalid rank\n");
        return 1;
    }

    int shape[MAX_RANK] = {0};
    if (rank > 0 && fread(shape, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        fclose(fp);
        printf("read shape failed\n");
        return 1;
    }

    float fill_value = 0.0f;
    if (fread(&fill_value, sizeof(float), 1, fp) != 1) {
        fclose(fp);
        printf("read fill value failed\n");
        return 1;
    }
    fclose(fp);

    for (int d = 0; d < rank; ++d) {
        if (shape[d] < 0) {
            printf("invalid shape\n");
            return 1;
        }
    }

    size_t expected_out_len = shape_size(shape, rank);
    if (out_len != expected_out_len) {
        printf("out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    size_t out_bytes = out_len * sizeof(float);
    float* h_output = (float*)malloc(out_bytes);
    if (!h_output) {
        printf("malloc failed\n");
        return 1;
    }

    float* d_output = NULL;
    cudaMalloc(&d_output, out_bytes);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    constant_of_shape_kernel<<<blocks, threads>>>(d_output, fill_value, out_len);
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

    cudaFree(d_output);
    free(h_output);
    return 0;
}
