/**
  ******************************************************************************
  * @file        verify_stft.cu
  * @author      Egor Izmaylov
  * @brief       提供 STFT 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 实现 `stft_kernel` CUDA 参考 kernel，按帧切片、可选窗口和朴素 DFT 公式生成频谱。
__global__ void stft_kernel(
    const double* signal,
    const double* window,
    double* Y,
    int batch,
    int signal_len,
    int signal_complex_dim,
    int n_frames,
    int bins,
    int frame_step,
    int frame_length,
    int has_window
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = batch * n_frames * bins * 2;
    if (idx >= total) return;

    int component = idx % 2;
    int t = idx / 2;
    int k = t % bins; t /= bins;
    int frame = t % n_frames;
    int b = t / n_frames;

    double real_sum = 0.0;
    double imag_sum = 0.0;
    for (int n = 0; n < frame_length; ++n) {
        int signal_pos = frame * frame_step + n;
        double xr = 0.0;
        double xi = 0.0;
        if (signal_pos >= 0 && signal_pos < signal_len) {
            size_t in_base = ((size_t)b * signal_len + signal_pos) * signal_complex_dim;
            xr = signal[in_base];
            xi = signal_complex_dim == 2 ? signal[in_base + 1] : 0.0;
        }
        double win = has_window ? window[n] : 1.0;
        xr *= win;
        xi *= win;

        double angle = -TWO_PI * (double)k * (double)n / (double)frame_length;
        double ca = cos(angle);
        double sa = sin(angle);
        real_sum += xr * ca - xi * sa;
        imag_sum += xr * sa + xi * ca;
    }

    Y[idx] = component == 0 ? real_sum : imag_sum;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <signal.bin> <frame_step.bin> <window.bin|null> <frame_length.bin> <params.bin> <out.bin>
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <out_len> <signal.bin> <frame_step.bin> <window.bin|null> <frame_length.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t p[9];
    FILE* fp = fopen(argv[6], "rb");
    if (!fp) return 2;
    if (fread(p, sizeof(int32_t), 9, fp) != 9) {
        fclose(fp);
        return 3;
    }
    fclose(fp);

    int batch = p[0], signal_len = p[1], signal_complex_dim = p[2];
    int n_frames = p[3], bins = p[4], frame_step = p[5], frame_length = p[6], has_window = p[8];
    if (out_len != (size_t)batch * n_frames * bins * 2) return 4;

    size_t signal_size = (size_t)batch * signal_len * signal_complex_dim;
    double* h_signal = (double*)malloc(signal_size * sizeof(double));
    double* h_window = NULL;
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_signal || !h_y) return 5;

    FILE* fs = fopen(argv[2], "rb");
    if (!fs) return 6;
    fread(h_signal, sizeof(double), signal_size, fs);
    fclose(fs);

    if (has_window) {
        h_window = (double*)malloc((size_t)frame_length * sizeof(double));
        if (!h_window) return 7;
        FILE* fw = fopen(argv[4], "rb");
        if (!fw) return 8;
        fread(h_window, sizeof(double), (size_t)frame_length, fw);
        fclose(fw);
    }

    double* d_signal = NULL;
    double* d_window = NULL;
    double* d_y = NULL;
    cudaMalloc(&d_signal, signal_size * sizeof(double));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMemcpy(d_signal, h_signal, signal_size * sizeof(double), cudaMemcpyHostToDevice);
    if (has_window) {
        cudaMalloc(&d_window, (size_t)frame_length * sizeof(double));
        cudaMemcpy(d_window, h_window, (size_t)frame_length * sizeof(double), cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    stft_kernel<<<blocks, threads>>>(
        d_signal,
        d_window,
        d_y,
        batch,
        signal_len,
        signal_complex_dim,
        n_frames,
        bins,
        frame_step,
        frame_length,
        has_window
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fo = fopen(argv[7], "wb");
    if (!fo) return 9;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_signal);
    if (h_window) free(h_window);
    free(h_y);
    cudaFree(d_signal);
    if (d_window) cudaFree(d_window);
    cudaFree(d_y);
    return 0;
}
