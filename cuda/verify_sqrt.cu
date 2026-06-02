#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Egor Izmaylov: Function `sqrt_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void sqrt_kernel(const float* a, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 主路径：输入非负
        out[idx] = sqrtf(a[idx]);
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <in0.bin> <out.bin>
    if (argc != 4) {
        printf("Usage: %s <out_len> <in0.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* in_path  = argv[2];
    const char* out_path = argv[3];

    size_t bytes = n * sizeof(float);

    float* h_a   = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fi = fopen(in_path, "rb");
    if (!fi) { printf("open input failed\n"); return 1; }
    size_t r = fread(h_a, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) { printf("fread mismatch\n"); return 1; }

    float *d_a = NULL, *d_out = NULL;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    sqrt_kernel<<<blocks, threads>>>(d_a, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_a);
    cudaFree(d_out);
    free(h_a);
    free(h_out);
    return 0;
}
