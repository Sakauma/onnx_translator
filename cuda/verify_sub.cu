#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sub_kernel_float32(const float* input_a, const float* input_b, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input_a[idx] - input_b[idx];
    }
}

int main(int argc, char** argv) {
    if (argc != 5) return 1;
    size_t num_elements = atol(argv[1]);
    size_t data_size = num_elements * sizeof(float);

    float *h_a = (float*)malloc(data_size);
    float *h_b = (float*)malloc(data_size);
    float *h_res = (float*)malloc(data_size);

    FILE* fp_a = fopen(argv[2], "rb"); fread(h_a, 1, data_size, fp_a); fclose(fp_a);
    FILE* fp_b = fopen(argv[3], "rb"); fread(h_b, 1, data_size, fp_b); fclose(fp_b);

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, data_size); cudaMalloc(&d_b, data_size); cudaMalloc(&d_out, data_size);
    cudaMemcpy(d_a, h_a, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, data_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    sub_kernel_float32<<<blocks, threads>>>(d_a, d_b, d_out, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_out, data_size, cudaMemcpyDeviceToHost);
    FILE* fp_out = fopen(argv[4], "wb"); fwrite(h_res, 1, data_size, fp_out); fclose(fp_out);

    free(h_a); free(h_b); free(h_res);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
}