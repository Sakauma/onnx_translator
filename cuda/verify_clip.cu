#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Egor Izmaylov: Function `clip_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void clip_kernel(
    const float* x,
    const float* min_v,
    const float* max_v,
    float* out,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float lo = min_v[0];
        float hi = max_v[0];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[idx] = v;
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <in0> <in1> <in2> <out>
    if (argc != 6) {
        printf("Usage: %s <out_len> <x.bin> <min.bin> <max.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* x_path   = argv[2];
    const char* min_path = argv[3];
    const char* max_path = argv[4];
    const char* out_path = argv[5];

    size_t bytes = n * sizeof(float);

    // ---- host ----
    float* h_x   = (float*)malloc(bytes);
    float* h_min = (float*)malloc(sizeof(float));
    float* h_max = (float*)malloc(sizeof(float));
    float* h_out = (float*)malloc(bytes);
    if (!h_x || !h_min || !h_max || !h_out) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* fx = fopen(x_path, "rb");
    FILE* fmin = fopen(min_path, "rb");
    FILE* fmax = fopen(max_path, "rb");
    if (!fx || !fmin || !fmax) {
        printf("failed to open input files\n");
        return 1;
    }

    size_t rx = fread(h_x, sizeof(float), n, fx);
    size_t rmin = fread(h_min, sizeof(float), 1, fmin);
    size_t rmax = fread(h_max, sizeof(float), 1, fmax);

    if (rx != n || rmin != 1 || rmax != 1) {
        printf("fread size mismatch\n");
        return 1;
    }
    fclose(fx);
    fclose(fmin);
    fclose(fmax);

    // ---- device ----
    float *d_x = NULL, *d_min = NULL, *d_max = NULL, *d_out = NULL;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, h_max, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    clip_kernel<<<blocks, threads>>>(d_x, d_min, d_max, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);

    cudaFree(d_x);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_out);
    free(h_x);
    free(h_min);
    free(h_max);
    free(h_out);
    return 0;
}
