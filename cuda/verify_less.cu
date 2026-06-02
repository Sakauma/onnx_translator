#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Egor Izmaylov: Function `less_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void less_kernel(const float* a, const float* b, unsigned char* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] < b[idx]) ? (unsigned char)1 : (unsigned char)0;
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: %s <out_len> <in0.bin> <in1.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* in0_path = argv[2];
    const char* in1_path = argv[3];
    const char* out_path = argv[4];

    size_t in_bytes = n * sizeof(float);
    size_t out_bytes = n * sizeof(unsigned char);

    float* h_a = (float*)malloc(in_bytes);
    float* h_b = (float*)malloc(in_bytes);
    unsigned char* h_out = (unsigned char*)malloc(out_bytes);
    if (!h_a || !h_b || !h_out) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* f0 = fopen(in0_path, "rb");
    FILE* f1 = fopen(in1_path, "rb");
    if (!f0 || !f1) {
        printf("open input failed\n");
        return 1;
    }
    size_t r0 = fread(h_a, sizeof(float), n, f0);
    size_t r1 = fread(h_b, sizeof(float), n, f1);
    if (r0 != n || r1 != n) {
        printf("fread size mismatch: r0=%zu r1=%zu expected=%zu\n", r0, r1, n);
        return 1;
    }
    fclose(f0);
    fclose(f1);

    float *d_a = NULL, *d_b = NULL;
    unsigned char* d_out = NULL;
    cudaMalloc(&d_a, in_bytes);
    cudaMalloc(&d_b, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_a, h_a, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    less_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        printf("open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_out, sizeof(unsigned char), n, fo);
    if (w != n) {
        printf("fwrite size mismatch: w=%zu expected=%zu\n", w, n);
        return 1;
    }
    fclose(fo);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    return 0;
}