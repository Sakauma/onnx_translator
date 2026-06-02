#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Egor Izmaylov: Function `max_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void max_kernel(const float* A, const float* B, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < n) {
        float a = A[idx];
        float b = B[idx];
        out[idx] = fmaxf(a,b);
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <n> <a.bin> <b.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <n> <a.bin> <b.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoll(argv[1]);
    size_t bytes = n * sizeof(float);

    float* h_a  = (float*)malloc(bytes);
    float* h_b  = (float*)malloc(bytes);
    float* h_out= (float*)malloc(bytes);
    if (!h_a || !h_b || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fa = fopen(argv[2], "rb");
    FILE* fb = fopen(argv[3], "rb");
    if (!fa || !fb) { fprintf(stderr, "open input failed\n"); return 1; }
    size_t ra = fread(h_a, sizeof(float), n, fa);
    size_t rb = fread(h_b, sizeof(float), n, fb);
    fclose(fa); fclose(fb);
    if (ra != n || rb != n) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_a=NULL, *d_b=NULL, *d_out=NULL;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    max_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[4], "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}
