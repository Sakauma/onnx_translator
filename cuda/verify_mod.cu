#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Egor Izmaylov: Function `python_mod` is a CUDA device helper used inside verifier kernels, keeping parameter decoding and saturation rules consistent across threads.
__device__ __forceinline__ float python_mod(float a, float b) {
    if (b == 0.0f) return NAN;
    float r = fmodf(a, b);
    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
    return r;
}

// Egor Izmaylov: Function `mod_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void mod_kernel(const float* a, const float* b, float* out, int n) {
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (t < n) out[t] = python_mod(a[t], b[t]);
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <a.bin> <b.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <a.bin> <b.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    int out_len = (int)atoll(argv[1]);
    const char* a_path = argv[2];
    const char* b_path = argv[3];
    const char* out_path = argv[4];

    size_t bytes = (size_t)out_len * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fa = fopen(a_path, "rb");
    FILE* fb = fopen(b_path, "rb");
    if (!fa || !fb) { printf("open input failed\n"); return 1; }

    size_t ra = fread(h_a, sizeof(float), out_len, fa);
    size_t rb = fread(h_b, sizeof(float), out_len, fb);
    fclose(fa); fclose(fb);
    if (ra != (size_t)out_len || rb != (size_t)out_len) { printf("fread mismatch\n"); return 1; }

    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    mod_kernel<<<blocks, threads>>>(d_a, d_b, d_out, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t wo = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != (size_t)out_len) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}