/*
 * 文件功能：提供 matmul integer 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ long long read_a_zp(const double* zp, int size, int row, int idx) {
    if (zp == NULL || size <= 0) return 0;
    if (size == 1) return llround(zp[0]);
    if (size > row) return llround(zp[row]);
    return llround(zp[idx]);
}

__device__ long long read_b_zp(const double* zp, int size, int col, int idx) {
    if (zp == NULL || size <= 0) return 0;
    if (size == 1) return llround(zp[0]);
    if (size > col) return llround(zp[col]);
    return llround(zp[idx]);
}

// 实现 `matmul_integer_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void matmul_integer_kernel(const double* A, const double* B,
                                      const double* AZeroPoint, const double* BZeroPoint,
                                      int32_t* Y,
                                      int M, int K, int N,
                                      int a_zp_size, int b_zp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int m = idx / N;
    int n = idx % N;
    long long acc = 0;
    for (int k = 0; k < K; k++) {
        int a_idx = m * K + k;
        int b_idx = k * N + n;
        long long a_val = llround(A[a_idx]);
        long long b_val = llround(B[b_idx]);
        long long a_zp = read_a_zp(AZeroPoint, a_zp_size, m, a_idx);
        long long b_zp = read_b_zp(BZeroPoint, b_zp_size, n, b_idx);
        acc += (a_val - a_zp) * (b_val - b_zp);
    }
    Y[idx] = (int32_t)acc;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 8) return 1;

    long long out_len = atoll(argv[1]);
    int p[5];
    FILE* fp = fopen(argv[6], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 5, fp);
    fclose(fp);

    int M = p[0], K = p[1], N = p[2];
    int a_zp_size = p[3], b_zp_size = p[4];
    size_t size_a = (size_t)M * K * sizeof(double);
    size_t size_b = (size_t)K * N * sizeof(double);
    size_t size_a_zp = (size_t)a_zp_size * sizeof(double);
    size_t size_b_zp = (size_t)b_zp_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(int32_t);

    double* h_a = (double*)malloc(size_a);
    double* h_b = (double*)malloc(size_b);
    double* h_a_zp = (double*)malloc(size_a_zp);
    double* h_b_zp = (double*)malloc(size_b_zp);
    int32_t* h_y = (int32_t*)malloc(size_y);

    FILE* f = fopen(argv[2], "rb"); fread(h_a, 1, size_a, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(h_b, 1, size_b, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(h_a_zp, 1, size_a_zp, f); fclose(f);
    f = fopen(argv[5], "rb"); fread(h_b_zp, 1, size_b_zp, f); fclose(f);

    double *d_a, *d_b, *d_a_zp, *d_b_zp;
    int32_t* d_y;
    cudaMalloc(&d_a, size_a); cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size_b); cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMalloc(&d_a_zp, size_a_zp); cudaMemcpy(d_a_zp, h_a_zp, size_a_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b_zp, size_b_zp); cudaMemcpy(d_b_zp, h_b_zp, size_b_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);

    matmul_integer_kernel<<<(out_len + 255) / 256, 256>>>(d_a, d_b, d_a_zp, d_b_zp, d_y, M, K, N, a_zp_size, b_zp_size);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    f = fopen(argv[7], "wb"); fwrite(h_y, 1, size_y, f); fclose(f);

    free(h_a); free(h_b); free(h_a_zp); free(h_b_zp); free(h_y);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_zp); cudaFree(d_b_zp); cudaFree(d_y);
    return 0;
}
