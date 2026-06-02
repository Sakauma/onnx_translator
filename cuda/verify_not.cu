#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Egor Izmaylov: Function `not_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void not_kernel(const float* in, unsigned char* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = (unsigned char)(x==0.0f ? 1 : 0);
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <n> <in.bin> <out.bin>
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <n> <in.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoll(argv[1]);
    size_t in_bytes = n * sizeof(float);
    size_t out_bytes = n * sizeof(unsigned char);

    float* h_in  = (float*)malloc(in_bytes);
    unsigned char* h_out = (unsigned char*)malloc(out_bytes);
    if (!h_in || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t r = fread(h_in, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_in = NULL; unsigned char* d_out = NULL;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    not_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(unsigned char), n, fo);
    fclose(fo);
    if (w != n) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
