/*
 * 文件功能：提供 clip 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 实现 `clip_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void clip_kernel(
    const float* x,
    const float* min_v,
    const float* max_v,
    float* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float lo = min_v[0];
        float hi = max_v[0];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[idx] = v;
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <in0> <in1> <in2> <out>
    if (argc != 6) {
        printf("Usage: %s <out_len> <x.bin> <min.bin> <max.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* x_path   = argv[2];
    const char* min_path = argv[3];
    const char* max_path = argv[4];
    const char* out_path = argv[5];

    size_t bytes = n * sizeof(float);

    // ---- host ----
    float* h_x   = (float*)malloc(bytes);
    float* h_min = (float*)malloc(sizeof(float));
    float* h_max = (float*)malloc(sizeof(float));
    float* h_out = (float*)malloc(bytes);
    if (!h_x || !h_min || !h_max || !h_out) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* fx = fopen(x_path, "rb");
    FILE* fmin = fopen(min_path, "rb");
    FILE* fmax = fopen(max_path, "rb");
    if (!fx || !fmin || !fmax) {
        printf("failed to open input files\n");
        return 1;
    }

    size_t rx = fread(h_x, sizeof(float), n, fx);
    size_t rmin = fread(h_min, sizeof(float), 1, fmin);
    size_t rmax = fread(h_max, sizeof(float), 1, fmax);

    if (rx != n || rmin != 1 || rmax != 1) {
        printf("fread size mismatch\n");
        return 1;
    }
    fclose(fx);
    fclose(fmin);
    fclose(fmax);

    // ---- device ----
    float *d_x = NULL, *d_min = NULL, *d_max = NULL, *d_out = NULL;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, h_max, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    clip_kernel<<<blocks, threads>>>(d_x, d_min, d_max, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);

    cudaFree(d_x);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_out);
    free(h_x);
    free(h_min);
    free(h_max);
    free(h_out);
    return 0;
}
