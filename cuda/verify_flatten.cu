/**
  ******************************************************************************
  * @file        verify_flatten.cu
  * @author      Egor Izmaylov
  * @brief       提供 flatten 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Flatten 只改变输出形状，不改变元素线性顺序；CUDA 参考按平坦顺序复制输入。
__global__ void copy_kernel(const float* input, float* output, size_t out_len) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < out_len) {
        output[t] = input[t];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <out.bin>
    if (argc != 4) {
        printf("Usage: %s <out_len> <input.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* out_path = argv[3];
    size_t bytes = out_len * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        printf("open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), out_len, fi);
    fclose(fi);
    if (ri != out_len) {
        printf("fread mismatch\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    copy_kernel<<<blocks, threads>>>(d_input, d_output, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

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
    free(h_input);
    free(h_output);
    return 0;
}
