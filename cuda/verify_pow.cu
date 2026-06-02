#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Egor Izmaylov: Function `pow_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void pow_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 主路径：float32 pow
        out[idx] = powf(a[idx], b[idx]);
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <数量> <输入A> <输入B> <输出>
    if (argc != 5) {
        printf("Usage: %s <out_len> <in0.bin> <in1.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* in0_path = argv[2];
    const char* in1_path = argv[3];
    const char* out_path = argv[4];

    size_t bytes = n * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_out) {
        printf("malloc failed\n");
        return 1;
    }

    FILE* f0 = fopen(in0_path, "rb");
    FILE* f1 = fopen(in1_path, "rb");
    if (!f0 || !f1) {
        printf("failed to open input files\n");
        return 1;
    }

    size_t r0 = fread(h_a, sizeof(float), n, f0);
    size_t r1 = fread(h_b, sizeof(float), n, f1);
    fclose(f0);
    fclose(f1);
    if (r0 != n || r1 != n) {
        printf("fread size mismatch\n");
        return 1;
    }

    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    pow_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        printf("failed to open output file\n");
        return 1;
    }
    size_t w = fwrite(h_out, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) {
        printf("fwrite size mismatch\n");
        return 1;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    return 0;
}
