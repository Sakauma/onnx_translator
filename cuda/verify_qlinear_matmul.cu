/**
  ******************************************************************************
  * @file        verify_qlinear_matmul.cu
  * @author      Egor Izmaylov
  * @brief       提供 qlinear matmul 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// 实现 `read_row_param` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ double read_row_param(const double* data, int size, int row, int idx, double default_value) {
    if (data == NULL || size <= 0) return default_value;
    if (size == 1) return data[0];
    if (size > row) return data[row];
    return data[idx];
}

// 实现 `read_col_param` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ double read_col_param(const double* data, int size, int col, int idx, double default_value) {
    if (data == NULL || size <= 0) return default_value;
    if (size == 1) return data[0];
    if (size > col) return data[col];
    return data[idx];
}

// 实现 `saturate_uint8` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ uint8_t saturate_uint8(double value) {
    long long rounded = llrint(value);
    if (rounded < 0) return 0;
    if (rounded > 255) return 255;
    return (uint8_t)rounded;
}

// 实现 `qlinear_matmul_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void qlinear_matmul_kernel(const double* A, const double* AScale, const double* AZeroPoint,
                                      const double* B, const double* BScale, const double* BZeroPoint,
                                      const double* YScale, const double* YZeroPoint, uint8_t* Y,
                                      int M, int K, int N,
                                      int a_scale_size, int a_zp_size,
                                      int b_scale_size, int b_zp_size,
                                      int y_scale_size, int y_zp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int m = idx / N;
    int n = idx % N;
    double acc = 0.0;
    for (int k = 0; k < K; k++) {
        int a_idx = m * K + k;
        int b_idx = k * N + n;
        double a_scale = read_row_param(AScale, a_scale_size, m, a_idx, 1.0);
        double a_zp = read_row_param(AZeroPoint, a_zp_size, m, a_idx, 0.0);
        double b_scale = read_col_param(BScale, b_scale_size, n, b_idx, 1.0);
        double b_zp = read_col_param(BZeroPoint, b_zp_size, n, b_idx, 0.0);
        double a_real = (llround(A[a_idx]) - llround(a_zp)) * a_scale;
        double b_real = (llround(B[b_idx]) - llround(b_zp)) * b_scale;
        acc += a_real * b_real;
    }

    double y_scale = (y_scale_size == 1) ? YScale[0] : YScale[idx];
    double y_zp = (y_zp_size == 1) ? YZeroPoint[0] : YZeroPoint[idx];
    double q = y_scale == 0.0 ? y_zp : acc / y_scale + y_zp;
    Y[idx] = saturate_uint8(q);
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 12) return 1;

    long long out_len = atoll(argv[1]);
    int p[9];
    FILE* fp = fopen(argv[10], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 9, fp);
    fclose(fp);

    int M = p[0], K = p[1], N = p[2];
    int a_scale_size = p[3], a_zp_size = p[4];
    int b_scale_size = p[5], b_zp_size = p[6];
    int y_scale_size = p[7], y_zp_size = p[8];
    size_t size_a = (size_t)M * K * sizeof(double);
    size_t size_b = (size_t)K * N * sizeof(double);
    size_t size_a_scale = (size_t)a_scale_size * sizeof(double);
    size_t size_a_zp = (size_t)a_zp_size * sizeof(double);
    size_t size_b_scale = (size_t)b_scale_size * sizeof(double);
    size_t size_b_zp = (size_t)b_zp_size * sizeof(double);
    size_t size_y_scale = (size_t)y_scale_size * sizeof(double);
    size_t size_y_zp = (size_t)y_zp_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(uint8_t);

    double* h_a = (double*)malloc(size_a);
    double* h_a_scale = (double*)malloc(size_a_scale);
    double* h_a_zp = (double*)malloc(size_a_zp);
    double* h_b = (double*)malloc(size_b);
    double* h_b_scale = (double*)malloc(size_b_scale);
    double* h_b_zp = (double*)malloc(size_b_zp);
    double* h_y_scale = (double*)malloc(size_y_scale);
    double* h_y_zp = (double*)malloc(size_y_zp);
    uint8_t* h_y = (uint8_t*)malloc(size_y);

    FILE* f = fopen(argv[2], "rb"); fread(h_a, 1, size_a, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(h_a_scale, 1, size_a_scale, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(h_a_zp, 1, size_a_zp, f); fclose(f);
    f = fopen(argv[5], "rb"); fread(h_b, 1, size_b, f); fclose(f);
    f = fopen(argv[6], "rb"); fread(h_b_scale, 1, size_b_scale, f); fclose(f);
    f = fopen(argv[7], "rb"); fread(h_b_zp, 1, size_b_zp, f); fclose(f);
    f = fopen(argv[8], "rb"); fread(h_y_scale, 1, size_y_scale, f); fclose(f);
    f = fopen(argv[9], "rb"); fread(h_y_zp, 1, size_y_zp, f); fclose(f);

    double *d_a, *d_a_scale, *d_a_zp, *d_b, *d_b_scale, *d_b_zp, *d_y_scale, *d_y_zp;
    uint8_t* d_y;
    cudaMalloc(&d_a, size_a); cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMalloc(&d_a_scale, size_a_scale); cudaMemcpy(d_a_scale, h_a_scale, size_a_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_a_zp, size_a_zp); cudaMemcpy(d_a_zp, h_a_zp, size_a_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size_b); cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b_scale, size_b_scale); cudaMemcpy(d_b_scale, h_b_scale, size_b_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b_zp, size_b_zp); cudaMemcpy(d_b_zp, h_b_zp, size_b_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y_scale, size_y_scale); cudaMemcpy(d_y_scale, h_y_scale, size_y_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y_zp, size_y_zp); cudaMemcpy(d_y_zp, h_y_zp, size_y_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);

    qlinear_matmul_kernel<<<(out_len + 255) / 256, 256>>>(d_a, d_a_scale, d_a_zp,
        d_b, d_b_scale, d_b_zp, d_y_scale, d_y_zp, d_y,
        M, K, N, a_scale_size, a_zp_size, b_scale_size, b_zp_size, y_scale_size, y_zp_size);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    f = fopen(argv[11], "wb"); fwrite(h_y, 1, size_y, f); fclose(f);

    free(h_a); free(h_a_scale); free(h_a_zp); free(h_b); free(h_b_scale);
    free(h_b_zp); free(h_y_scale); free(h_y_zp); free(h_y);
    cudaFree(d_a); cudaFree(d_a_scale); cudaFree(d_a_zp); cudaFree(d_b); cudaFree(d_b_scale);
    cudaFree(d_b_zp); cudaFree(d_y_scale); cudaFree(d_y_zp); cudaFree(d_y);
    return 0;
}
