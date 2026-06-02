/*
 * 文件功能：提供 global max pool 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

// 实现 `global_max_pool_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void global_max_pool_kernel(const double* X, double* Y, int outer, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer) return;

    double max_val = -DBL_MAX;
    int offset = idx * spatial_size;
    for (int i = 0; i < spatial_size; i++) {
        double val = X[offset + i];
        if (val > max_val) max_val = val;
    }
    Y[idx] = max_val;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 5) return 1;

    long long out_len = atoll(argv[1]);
    int p[3];
    FILE* fp = fopen(argv[3], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 3, fp);
    fclose(fp);

    int outer = p[0] * p[1];
    int spatial_size = p[2];
    size_t size_x = (size_t)outer * spatial_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(double);
    double* h_x = (double*)malloc(size_x);
    double* h_y = (double*)malloc(size_y);

    FILE* fx = fopen(argv[2], "rb");
    if (!fx) return 3;
    fread(h_x, 1, size_x, fx);
    fclose(fx);

    double *d_x, *d_y;
    cudaMalloc(&d_x, size_x);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    global_max_pool_kernel<<<(out_len + 255) / 256, 256>>>(d_x, d_y, outer, spatial_size);
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

    FILE* fout = fopen(argv[4], "wb");
    if (!fout) return 4;
    fwrite(h_y, 1, size_y, fout);
    fclose(fout);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
