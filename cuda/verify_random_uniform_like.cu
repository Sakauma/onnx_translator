#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

struct RandomUniformLikeParams {
    int32_t numel;
    float low;
    float high;
    uint32_t seed;
};

// Egor Izmaylov: Function `lcg_next` is a CUDA device helper used inside verifier kernels, keeping parameter decoding and saturation rules consistent across threads.
__device__ __forceinline__ uint32_t lcg_next(uint32_t x) {
    return x * 1664525u + 1013904223u;
}

// Egor Izmaylov: Function `random_uniform_like_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void random_uniform_like_kernel(float* out, int numel, float low, float high, uint32_t seed) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= numel) return;

    uint32_t s = seed ^ (uint32_t)tid;
    s = lcg_next(s);

    // 取低 24 bit 映射到 [0,1)
    float u = (float)(s & 0x00FFFFFFu) / 16777216.0f;
    out[tid] = low + (high - low) * u;
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <dummy_input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <dummy_input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* p_path = argv[3];
    const char* out_path = argv[4];

    RandomUniformLikeParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(RandomUniformLikeParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if ((size_t)p.numel != out_len) {
        fprintf(stderr, "out_len mismatch: got %zu expected %d\n", out_len, p.numel);
        return 1;
    }

    float* d_out = NULL;
    float* h_out = (float*)malloc(out_len * sizeof(float));
    if (!h_out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    cudaMalloc(&d_out, out_len * sizeof(float));

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    random_uniform_like_kernel<<<blocks, threads>>>(d_out, p.numel, p.low, p.high, p.seed);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    if (fwrite(h_out, sizeof(float), out_len, fo) != out_len) {
        fprintf(stderr, "fwrite mismatch\n");
        fclose(fo);
        return 1;
    }
    fclose(fo);

    cudaFree(d_out);
    free(h_out);
    return 0;
}