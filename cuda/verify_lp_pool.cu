/*
 * 文件功能：提供 lp pool 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 实现 `lp_pool_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void lp_pool_kernel(const double* X, double* Y,
                               int batch, int channels, int in_h, int in_w,
                               int out_h, int out_w,
                               int k_h, int k_w,
                               int pad_t, int pad_l,
                               int stride_h, int stride_w,
                               int dil_h, int dil_w,
                               int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int c = temp % channels; temp /= channels;
    int n = temp;

    double sum_pow = 0.0;
    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            int ih = oh * stride_h + kh * dil_h - pad_t;
            int iw = ow * stride_w + kw * dil_w - pad_l;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int x_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                sum_pow += pow(fabs(X[x_idx]), (double)p);
            }
        }
    }
    Y[idx] = pow(sum_pow, 1.0 / (double)p);
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 5) return 1;

    long long out_len = atoll(argv[1]);
    int p[15];
    FILE* fp = fopen(argv[3], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 15, fp);
    fclose(fp);

    size_t size_x = (size_t)p[0] * p[1] * p[2] * p[3] * sizeof(double);
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

    lp_pool_kernel<<<(out_len + 255) / 256, 256>>>(d_x, d_y,
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        p[8], p[9], p[10], p[11], p[12], p[13], p[14]);

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
