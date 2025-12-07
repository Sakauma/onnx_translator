#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ double saturate_cast_int8(double val) {
    if (val > 127.0) return 127.0;
    if (val < -128.0) return -128.0;
    return val;
}

__device__ double saturate_cast_uint8(double val) {
    if (val > 255.0) return 255.0;
    if (val < 0.0) return 0.0;
    return val;
}

__global__ void quantize_kernel(const double* x, const double* scale, const double* zp, double* out, size_t n, int is_signed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double s = scale[idx];
        double z = zp[idx];
        double res = rint(x[idx] / s) + z;
        
        if (is_signed) {
            out[idx] = saturate_cast_int8(res);
        } else {
            out[idx] = saturate_cast_uint8(res);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 7) return 1; 
    size_t n = atol(argv[1]);
    size_t bytes = n * sizeof(double);
    
    int is_signed = 1;
    FILE *fp = fopen(argv[5], "rb");
    if (fp) {
        fread(&is_signed, sizeof(int), 1, fp);
        fclose(fp);
    }

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

    quantize_kernel<<<(n + 255)/256, 256>>>(d_x, d_s, d_z, d_out, n, is_signed);
    
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[6], "wb"); fwrite(h_out, 1, bytes, fout); fclose(fout);
    
    free(h_x); free(h_s); free(h_z); free(h_out);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_z); cudaFree(d_out);
    return 0;
}