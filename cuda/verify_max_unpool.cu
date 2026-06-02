/*
 * 文件功能：提供 max unpool 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

// 实现 `max_unpool_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void max_unpool_kernel(const double* X, const int64_t* Indices, double* Y,
                                  int input_size, int inferred_total,
                                  int channels, int inferred_h, int inferred_w,
                                  int out_h, int out_w) {
    int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_idx >= input_size) return;

    int64_t flat_index = Indices[src_idx];
    if (flat_index < 0 || flat_index >= inferred_total) return;

    int n = (int)(flat_index / (channels * inferred_h * inferred_w));
    int rem = (int)(flat_index % (channels * inferred_h * inferred_w));
    int c = rem / (inferred_h * inferred_w);
    rem %= inferred_h * inferred_w;
    int h = rem / inferred_w;
    int w = rem % inferred_w;

    if (h >= out_h || w >= out_w) return;
    int dst_idx = ((n * channels + c) * out_h + h) * out_w + w;
    Y[dst_idx] = X[src_idx];
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 5) return 1;

    long long out_len = atoll(argv[1]);
    int p[14];
    FILE* fp = fopen(argv[4], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 14, fp);
    fclose(fp);

    int N = p[0], C = p[1], IH = p[2], IW = p[3];
    int OH = p[4], OW = p[5];
    int KH = p[6], KW = p[7];
    int pad_t = p[8], pad_l = p[9], pad_b = p[10], pad_r = p[11];
    int stride_h = p[12], stride_w = p[13];
    int inferred_h = (IH - 1) * stride_h - pad_t - pad_b + KH;
    int inferred_w = (IW - 1) * stride_w - pad_l - pad_r + KW;
    int input_size = N * C * IH * IW;
    int inferred_total = N * C * inferred_h * inferred_w;

    size_t size_x = (size_t)input_size * sizeof(double);
    size_t size_i = (size_t)input_size * sizeof(int64_t);
    size_t size_y = (size_t)out_len * sizeof(double);

    double* h_x = (double*)malloc(size_x);
    int64_t* h_i = (int64_t*)malloc(size_i);
    double* h_y = (double*)calloc((size_t)out_len, sizeof(double));

    FILE* fx = fopen(argv[2], "rb"); fread(h_x, 1, size_x, fx); fclose(fx);
    FILE* fi = fopen(argv[3], "rb"); fread(h_i, 1, size_i, fi); fclose(fi);

    double *d_x, *d_y;
    int64_t* d_i;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_i, size_i); cudaMemcpy(d_i, h_i, size_i, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y); cudaMemset(d_y, 0, size_y);

    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    max_unpool_kernel<<<blocks, threads>>>(d_x, d_i, d_y, input_size, inferred_total, C, inferred_h, inferred_w, OH, OW);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE* fout = fopen(argv[5], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);

    free(h_x); free(h_i); free(h_y);
    cudaFree(d_x); cudaFree(d_i); cudaFree(d_y);
    return 0;
}
