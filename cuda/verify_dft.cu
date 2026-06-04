/**
  ******************************************************************************
  * @file        verify_dft.cu
  * @author      Egor Izmaylov
  * @brief       提供 DFT 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.04  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TWO_PI 6.283185307179586476925286766559

// 实现 `dft_kernel` CUDA 参考 kernel，覆盖实数或复数输入的前向/逆向 DFT 主路径。
__global__ void dft_kernel(
    const double* X,
    double* Y,
    int batch,
    int input_len,
    int input_complex_dim,
    int output_len,
    int output_complex_dim,
    int inverse,
    int onesided,
    int dft_length
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = batch * output_len * output_complex_dim;
    if (idx >= total) return;

    int component = idx % output_complex_dim;
    int t = idx / output_complex_dim;
    int k = t % output_len;
    int b = t / output_len;

    if (inverse && onesided) {
        double real_sum = 0.0;
        int max_freq = dft_length / 2;
        for (int f = 0; f < input_len; ++f) {
            if (f > max_freq) continue;
            size_t in_base = ((size_t)b * input_len + f) * input_complex_dim;
            double xr = X[in_base];
            double xi = input_complex_dim == 2 ? X[in_base + 1] : 0.0;
            double angle = TWO_PI * (double)f * (double)k / (double)dft_length;
            double contribution = xr * cos(angle) - xi * sin(angle);
            if (f != 0 && !(dft_length % 2 == 0 && f == dft_length / 2)) {
                contribution *= 2.0;
            }
            real_sum += contribution;
        }
        Y[idx] = real_sum / (double)dft_length;
        return;
    }

    double real_sum = 0.0;
    double imag_sum = 0.0;
    double sign = inverse ? 1.0 : -1.0;
    for (int n = 0; n < dft_length; ++n) {
        double xr = 0.0;
        double xi = 0.0;
        if (n < input_len) {
            size_t in_base = ((size_t)b * input_len + n) * input_complex_dim;
            xr = X[in_base];
            xi = input_complex_dim == 2 ? X[in_base + 1] : 0.0;
        }
        double angle = sign * TWO_PI * (double)k * (double)n / (double)dft_length;
        double ca = cos(angle);
        double sa = sin(angle);
        real_sum += xr * ca - xi * sa;
        imag_sum += xr * sa + xi * ca;
    }

    if (inverse) {
        real_sum /= (double)dft_length;
        imag_sum /= (double)dft_length;
    }
    Y[idx] = component == 0 ? real_sum : imag_sum;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <X.bin> <dft_length.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <X.bin> <dft_length.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t p[9];
    FILE* fp = fopen(argv[4], "rb");
    if (!fp) return 2;
    if (fread(p, sizeof(int32_t), 9, fp) != 9) {
        fclose(fp);
        return 3;
    }
    fclose(fp);

    int batch = p[0], input_len = p[1], input_complex_dim = p[2];
    int output_len = p[3], output_complex_dim = p[4];
    int inverse = p[6], onesided = p[7], dft_length = p[8];
    if (out_len != (size_t)batch * output_len * output_complex_dim) return 4;

    size_t x_len = (size_t)batch * input_len * input_complex_dim;
    double* h_x = (double*)malloc(x_len * sizeof(double));
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_x || !h_y) return 5;

    FILE* fx = fopen(argv[2], "rb");
    if (!fx) return 6;
    fread(h_x, sizeof(double), x_len, fx);
    fclose(fx);

    double* d_x = NULL;
    double* d_y = NULL;
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    dft_kernel<<<blocks, threads>>>(d_x, d_y, batch, input_len, input_complex_dim, output_len, output_complex_dim, inverse, onesided, dft_length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fo = fopen(argv[5], "wb");
    if (!fo) return 7;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
