/**
  ******************************************************************************
  * @file        verify_dynamic_quantize_linear.cu
  * @author      Egor Izmaylov
  * @brief       提供 DynamicQuantizeLinear 算子的 CUDA 参考验证程序。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct DynamicQuantParams {
    float scale;
    float zero_point;
};

__global__ void reduce_minmax_kernel(const float* input, float* block_min, float* block_max, size_t n) {
    extern __shared__ float shared[];
    float* s_min = shared;
    float* s_max = shared + blockDim.x;

    unsigned int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;

    float v = 0.0f;
    if (idx < n) {
        v = input[idx];
    }
    s_min[tid] = idx < n ? v : FLT_MAX;
    s_max[tid] = idx < n ? v : -FLT_MAX;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = s_min[0];
        block_max[blockIdx.x] = s_max[0];
    }
}

__global__ void dynamic_quantize_kernel(
    const float* input,
    float* output,
    size_t n,
    DynamicQuantParams params
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x;
    if (tid >= n) return;

    float q = nearbyintf(input[tid] / params.scale) + params.zero_point;
    q = fminf(255.0f, fmaxf(0.0f, q));
    output[tid] = q;
}

static int read_f32_file(const char* path, float* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        return 0;
    }
    size_t got = fread(data, sizeof(float), n, fp);
    fclose(fp);
    return got == n;
}

static int write_f32_file(const char* path, const float* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        return 0;
    }
    size_t wrote = fwrite(data, sizeof(float), n, fp);
    fclose(fp);
    return wrote == n;
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;

    size_t out_len = (size_t)atoll(argv[1]);
    if (out_len < 3) return 1;
    size_t n = out_len - 2;

    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)calloc(out_len, sizeof(float));
    if (!h_x || !h_out) return 1;
    if (!read_f32_file(argv[2], h_x, n)) return 1;

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    float* d_x = NULL;
    float* d_out = NULL;
    float* d_block_min = NULL;
    float* d_block_max = NULL;
    float* h_block_min = (float*)malloc(blocks * sizeof(float));
    float* h_block_max = (float*)malloc(blocks * sizeof(float));
    if (!h_block_min || !h_block_max) return 1;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_block_min, blocks * sizeof(float));
    cudaMalloc(&d_block_max, blocks * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_minmax_kernel<<<blocks, threads, threads * 2 * sizeof(float)>>>(d_x, d_block_min, d_block_max, n);
    cudaMemcpy(h_block_min, d_block_min, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_max, d_block_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (int i = 0; i < blocks; ++i) {
        min_val = fminf(min_val, h_block_min[i]);
        max_val = fmaxf(max_val, h_block_max[i]);
    }
    min_val = fminf(min_val, 0.0f);
    max_val = fmaxf(max_val, 0.0f);

    DynamicQuantParams params;
    params.scale = (max_val - min_val) / 255.0f;
    if (params.scale == 0.0f) {
        params.scale = 1.0f;
    }
    float zp = roundf(0.0f - min_val / params.scale);
    zp = fminf(255.0f, fmaxf(0.0f, zp));
    params.zero_point = zp;

    dynamic_quantize_kernel<<<blocks, threads>>>(d_x, d_out, n, params);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[n] = params.scale;
    h_out[n + 1] = params.zero_point;

    int ok = write_f32_file(argv[3], h_out, out_len);

    free(h_x);
    free(h_out);
    free(h_block_min);
    free(h_block_max);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_block_min);
    cudaFree(d_block_max);
    return ok ? 0 : 1;
}
