#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Egor Izmaylov: Function `floor_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void floor_kernel(const float* in, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = floorf(x);
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
    size_t bytes = n * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t r = fread(h_in, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    floor_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
