#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

// Egor Izmaylov: Function `global_max_pool_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void global_max_pool_kernel(const double* X, double* Y, int outer, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer) return;

    double max_val = -DBL_MAX;
    int offset = idx * spatial_size;
    for (int i = 0; i < spatial_size; i++) {
        double val = X[offset + i];
        if (val > max_val) max_val = val;
    }
    Y[idx] = max_val;
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    if (argc < 5) return 1;

    long long out_len = atoll(argv[1]);
    int p[3];
    FILE* fp = fopen(argv[3], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 3, fp);
    fclose(fp);

    int outer = p[0] * p[1];
    int spatial_size = p[2];
    size_t size_x = (size_t)outer * spatial_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(double);
    double* h_x = (double*)malloc(size_x);
    double* h_y = (double*)malloc(size_y);

    FILE* fx = fopen(argv[2], "rb");
    if (!fx) return 3;
    fread(h_x, 1, size_x, fx);
    fclose(fx);

    double *d_x, *d_y;
    cudaMalloc(&d_x, size_x);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    global_max_pool_kernel<<<(out_len + 255) / 256, 256>>>(d_x, d_y, outer, spatial_size);
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

    FILE* fout = fopen(argv[4], "wb");
    if (!fout) return 4;
    fwrite(h_y, 1, size_y, fout);
    fclose(fout);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
