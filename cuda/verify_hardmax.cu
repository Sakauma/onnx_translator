/**
  ******************************************************************************
  * @file        verify_hardmax.cu
  * @author      Egor Izmaylov
  * @brief       提供 hardmax 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

// 按指定 axis 的 outer/inner/remaining 展开方式计算 one-hot hardmax。
__global__ void hardmax_kernel(const double* X, double* Y, int outer, int inner, int remaining) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * remaining) return;

    int o = idx / remaining;
    int r = idx % remaining;

    double max_val = -DBL_MAX;
    int max_idx = 0;
    for (int i = 0; i < inner; i++) {
        int x_idx = o * inner * remaining + i * remaining + r;
        if (X[x_idx] > max_val) {
            max_val = X[x_idx];
            max_idx = i;
        }
    }

    for (int i = 0; i < inner; i++) {
        int x_idx = o * inner * remaining + i * remaining + r;
        Y[x_idx] = (i == max_idx) ? 1.0 : 0.0;
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // argv: [1]size, [2]X, [3]params, [4]Y
    if (argc < 5) return 1;
    long long len = atoll(argv[1]);

    int p[3];
    FILE *fp = fopen(argv[3], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 3, fp);
    fclose(fp);

    size_t bytes = len * sizeof(double);
    double *h_x = (double*)malloc(bytes);
    double *h_y = (double*)malloc(bytes);

    FILE *fx = fopen(argv[2], "rb");
    fread(h_x, 1, bytes, fx);
    fclose(fx);

    double *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, bytes);

    int work_items = p[0] * p[2];
    hardmax_kernel<<<(work_items + 255) / 256, 256>>>(d_x, d_y, p[0], p[1], p[2]);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[4], "wb");
    fwrite(h_y, 1, bytes, fout);
    fclose(fout);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
