#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void relu_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cosf(in[idx]);
}

int main(int argc, char** argv) {
    if (argc != 4) return 1; 
    size_t n = atol(argv[1]);
    size_t bytes = n * sizeof(float);
    
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    
    FILE *fin = fopen(argv[2], "rb"); fread(h_in, 1, bytes, fin); fclose(fin);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    relu_kernel<<<(n + 255)/256, 256>>>(d_in, d_out, n);
    
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[3], "wb"); fwrite(h_out, 1, bytes, fout); fclose(fout);
    
    free(h_in); free(h_out); cudaFree(d_in); cudaFree(d_out);
    return 0;
}