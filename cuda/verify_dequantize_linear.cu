#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 升级: 所有指针和计算改为 double
__global__ void dequantize_kernel(const double* x, const double* scale, const double* zp, double* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (x[idx] - zp[idx]) * scale[idx];
    }
}

int main(int argc, char** argv) {
    if (argc != 6) return 1; 
    size_t n = atol(argv[1]);
    size_t bytes = n * sizeof(double);
    
    double *h_x = (double*)malloc(bytes);
    double *h_s = (double*)malloc(bytes);
    double *h_z = (double*)malloc(bytes);
    double *h_out = (double*)malloc(bytes);
    
    FILE *fx = fopen(argv[2], "rb"); fread(h_x, 1, bytes, fx); fclose(fx);
    FILE *fs = fopen(argv[3], "rb"); fread(h_s, 1, bytes, fs); fclose(fs);
    FILE *fz = fopen(argv[4], "rb"); fread(h_z, 1, bytes, fz); fclose(fz);
    
    double *d_x, *d_s, *d_z, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_s, bytes); cudaMalloc(&d_z, bytes); cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice);
    
    dequantize_kernel<<<(n + 255)/256, 256>>>(d_x, d_s, d_z, d_out, n);
    
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[5], "wb"); fwrite(h_out, 1, bytes, fout); fclose(fout);
    
    free(h_x); free(h_s); free(h_z); free(h_out);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_z); cudaFree(d_out);
    return 0;
}