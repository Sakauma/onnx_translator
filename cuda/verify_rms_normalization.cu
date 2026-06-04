/**
  ******************************************************************************
  * @file        verify_rms_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 RMSNormalization 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

struct RMSNormParams {
    int32_t row_count;
    int32_t normalized_size;
    float epsilon;
};

// 实现 `rms_normalization_kernel` CUDA 参考 kernel，按 axis 后缀分段计算 RMS 并应用已广播 scale。
__global__ void rms_normalization_kernel(const float* x, const float* scale, float* out, RMSNormParams p, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int row = idx / p.normalized_size;
    int offset = row * p.normalized_size;
    float square_sum = 0.0f;
    for (int j = 0; j < p.normalized_size; ++j) {
        float v = x[offset + j];
        square_sum += v * v;
    }
    float inv_rms = rsqrtf(square_sum / (float)p.normalized_size + p.epsilon);
    out[idx] = x[idx] * inv_rms * scale[idx];
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <x.bin> <scale.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <scale.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    int total = atoi(argv[1]);
    const char* x_path = argv[2];
    const char* scale_path = argv[3];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    RMSNormParams params;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    size_t pr = fread(&params, sizeof(RMSNormParams), 1, fp);
    fclose(fp);
    if (pr != 1 || params.row_count <= 0 || params.normalized_size <= 0) {
        fprintf(stderr, "read params failed\n");
        return 1;
    }

    float* h_x = (float*)malloc((size_t)total * sizeof(float));
    float* h_scale = (float*)malloc((size_t)total * sizeof(float));
    float* h_out = (float*)malloc((size_t)total * sizeof(float));
    if (!h_x || !h_scale || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        free(h_x);
        free(h_scale);
        free(h_out);
        return 1;
    }

    fp = fopen(x_path, "rb");
    if (!fp || fread(h_x, sizeof(float), total, fp) != (size_t)total) {
        fprintf(stderr, "read input failed\n");
        if (fp) fclose(fp);
        free(h_x);
        free(h_scale);
        free(h_out);
        return 1;
    }
    fclose(fp);

    fp = fopen(scale_path, "rb");
    if (!fp || fread(h_scale, sizeof(float), total, fp) != (size_t)total) {
        fprintf(stderr, "read scale failed\n");
        if (fp) fclose(fp);
        free(h_x);
        free(h_scale);
        free(h_out);
        return 1;
    }
    fclose(fp);

    float *d_x = NULL, *d_scale = NULL, *d_out = NULL;
    cudaMalloc((void**)&d_x, (size_t)total * sizeof(float));
    cudaMalloc((void**)&d_scale, (size_t)total * sizeof(float));
    cudaMalloc((void**)&d_out, (size_t)total * sizeof(float));
    cudaMemcpy(d_x, h_x, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rms_normalization_kernel<<<blocks, threads>>>(d_x, d_scale, d_out, params, total);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, (size_t)total * sizeof(float), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_x);
        cudaFree(d_scale);
        cudaFree(d_out);
        free(h_x);
        free(h_scale);
        free(h_out);
        return 1;
    }
    fwrite(h_out, sizeof(float), total, fp);
    fclose(fp);

    cudaFree(d_x);
    cudaFree(d_scale);
    cudaFree(d_out);
    free(h_x);
    free(h_scale);
    free(h_out);
    return 0;
}
