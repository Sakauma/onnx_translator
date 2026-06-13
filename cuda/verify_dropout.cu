/**
  ******************************************************************************
  * @file        verify_dropout.cu
  * @author      Egor Izmaylov
  * @brief       提供 Dropout 算子的 CUDA 参考验证程序，输出 y 和 mask。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MT_N 624
#define MT_M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

struct DropoutParams {
    int32_t input_len;
    int32_t training_mode;
    uint32_t seed;
    float ratio;
};

struct MTState {
    uint32_t mt[MT_N];
    int index;
};

__device__ void mt_seed(MTState* state, uint32_t seed) {
    state->mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        uint32_t prev = state->mt[i - 1];
        state->mt[i] = (uint32_t)(1812433253UL * (prev ^ (prev >> 30)) + (uint32_t)i);
    }
    state->index = MT_N;
}

__device__ uint32_t mt_uint32(MTState* state) {
    static const uint32_t mag01[2] = {0x0UL, MATRIX_A};

    if (state->index >= MT_N) {
        int kk = 0;
        for (; kk < MT_N - MT_M; ++kk) {
            uint32_t y = (state->mt[kk] & UPPER_MASK) | (state->mt[kk + 1] & LOWER_MASK);
            state->mt[kk] = state->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; ++kk) {
            uint32_t y = (state->mt[kk] & UPPER_MASK) | (state->mt[kk + 1] & LOWER_MASK);
            state->mt[kk] = state->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        uint32_t y = (state->mt[MT_N - 1] & UPPER_MASK) | (state->mt[0] & LOWER_MASK);
        state->mt[MT_N - 1] = state->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        state->index = 0;
    }

    uint32_t y = state->mt[state->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

__device__ double mt_uniform(MTState* state) {
    uint32_t a = mt_uint32(state) >> 5;
    uint32_t b = mt_uint32(state) >> 6;
    return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

__global__ void dropout_kernel(const float* input, float* output, uint8_t* mask, DropoutParams params) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    if (!params.training_mode || params.ratio == 0.0f) {
        for (int i = 0; i < params.input_len; ++i) {
            output[i] = input[i];
            mask[i] = 1;
        }
        return;
    }

    MTState state;
    mt_seed(&state, params.seed);
    float scale = 1.0f / (1.0f - params.ratio);
    for (int i = 0; i < params.input_len; ++i) {
        int keep = mt_uniform(&state) >= (double)params.ratio;
        mask[i] = (uint8_t)keep;
        output[i] = keep ? input[i] * scale : 0.0f;
    }
}

static int read_params(const char* path, DropoutParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int ok = fread(params, sizeof(DropoutParams), 1, fp) == 1;
    fclose(fp);
    return ok;
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

static int write_u8_file(const char* path, const uint8_t* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open mask failed\n");
        return 0;
    }
    size_t wrote = fwrite(data, sizeof(uint8_t), n, fp);
    fclose(fp);
    return wrote == n;
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;

    DropoutParams params;
    if (!read_params(argv[3], &params)) return 1;
    if (params.input_len < 0 || params.ratio < 0.0f || params.ratio >= 1.0f) return 1;

    size_t n = (size_t)params.input_len;
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));
    uint8_t* h_mask = (uint8_t*)malloc(n * sizeof(uint8_t));
    if (!h_x || !h_y || !h_mask) return 1;
    if (!read_f32_file(argv[2], h_x, n)) return 1;

    float* d_x = NULL;
    float* d_y = NULL;
    uint8_t* d_mask = NULL;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_mask, n * sizeof(uint8_t));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    dropout_kernel<<<1, 1>>>(d_x, d_y, d_mask, params);
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask, d_mask, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    int ok = write_f32_file(argv[4], h_y, n);
    ok = ok && write_u8_file("tmp_dropout_mask.bin", h_mask, n);

    free(h_x);
    free(h_y);
    free(h_mask);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_mask);
    return ok ? 0 : 1;
}
