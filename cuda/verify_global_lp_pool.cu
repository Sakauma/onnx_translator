#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void global_lp_pool_kernel(const double* X, double* Y, int outer, int spatial_size, int p_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer) return;

    double sum_pow = 0.0;
    int offset = idx * spatial_size;
    for (int i = 0; i < spatial_size; i++) {
        sum_pow += pow(fabs(X[offset + i]), (double)p_norm);
    }
    Y[idx] = pow(sum_pow, 1.0 / (double)p_norm);
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;

    long long out_len = atoll(argv[1]);
    int p[4];
    FILE* fp = fopen(argv[3], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 4, fp);
    fclose(fp);

    int outer = p[0] * p[1];
    int spatial_size = p[2];
    int p_norm = p[3];
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
    global_lp_pool_kernel<<<(out_len + 255) / 256, 256>>>(d_x, d_y, outer, spatial_size, p_norm);
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
