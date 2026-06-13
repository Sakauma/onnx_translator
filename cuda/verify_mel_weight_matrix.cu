/**
  ******************************************************************************
  * @file        verify_mel_weight_matrix.cu
  * @author      Egor Izmaylov
  * @brief       提供 MelWeightMatrix 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct MelParams {
    int32_t bins;
    int32_t dft_len;
    int32_t sample_rate;
    int32_t spectrogram_bins;
    float lower;
    float upper;
};

__device__ double hz_to_mel_device(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

__device__ double mel_to_hz_device(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

__global__ void mel_weight_matrix_kernel(float* output, MelParams p, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    int j = (int)(tid / (size_t)p.bins);
    int i = (int)(tid % (size_t)p.bins);
    double mel_lower = hz_to_mel_device((double)p.lower);
    double mel_upper = hz_to_mel_device((double)p.upper);
    double mel_step = (mel_upper - mel_lower) / (double)(p.bins + 2);

    double left_mel = mel_lower + mel_step * (double)i;
    double center_mel = mel_lower + mel_step * (double)(i + 1);
    double right_mel = mel_lower + mel_step * (double)(i + 2);

    int left = (int)floor((double)(p.dft_len + 1) * mel_to_hz_device(left_mel) / (double)p.sample_rate);
    int center = (int)floor((double)(p.dft_len + 1) * mel_to_hz_device(center_mel) / (double)p.sample_rate);
    int right = (int)floor((double)(p.dft_len + 1) * mel_to_hz_device(right_mel) / (double)p.sample_rate);

    if (left < 0) left = 0;
    if (center < 0) center = 0;
    if (center > p.spectrogram_bins - 1) center = p.spectrogram_bins - 1;
    if (right < 0) right = 0;
    if (right > p.spectrogram_bins) right = p.spectrogram_bins;

    double value = 0.0;
    if (center == left && j == center && center >= 0 && center < p.spectrogram_bins) {
        value = 1.0;
    } else if (j >= left && j <= center && j < p.spectrogram_bins && center != left) {
        value = (double)(j - left) / (double)(center - left);
    }
    if (right > center && j >= center && j < right && j < p.spectrogram_bins) {
        value = (double)(right - j) / (double)(right - center);
    }

    output[tid] = (float)value;
}

int main(int argc, char** argv) {
    // <out_len> <num_mel_bins.bin> <dft_length.bin> <sample_rate.bin> <lower.bin> <upper.bin> <params.bin> <out.bin>
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <out_len> <num_mel_bins.bin> <dft_length.bin> <sample_rate.bin> <lower.bin> <upper.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[7];
    const char* out_path = argv[8];

    int32_t int_params[4] = {0, 0, 0, 0};
    float float_params[2] = {0.0f, 0.0f};
    FILE* fp = fopen(params_path, "rb");
    if (!fp || fread(int_params, sizeof(int32_t), 4, fp) != 4 || fread(float_params, sizeof(float), 2, fp) != 2) {
        if (fp) fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 1;
    }
    fclose(fp);

    MelParams params;
    params.bins = int_params[0];
    params.dft_len = int_params[1];
    params.sample_rate = int_params[2];
    params.spectrogram_bins = int_params[3];
    params.lower = float_params[0];
    params.upper = float_params[1];
    if (params.bins <= 0 || params.spectrogram_bins <= 0 || out_len != (size_t)params.bins * (size_t)params.spectrogram_bins) {
        fprintf(stderr, "invalid params\n");
        return 1;
    }

    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    float* d_output = NULL;
    cudaMalloc(&d_output, out_len * sizeof(float));
    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    mel_weight_matrix_kernel<<<blocks, threads>>>(d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo || fwrite(h_output, sizeof(float), out_len, fo) != out_len) {
        if (fo) fclose(fo);
        fprintf(stderr, "write output failed\n");
        return 1;
    }
    fclose(fo);

    cudaFree(d_output);
    free(h_output);
    return 0;
}
