// verify.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "tensor_ops.h"
#include <math.h>

__global__ void cosine_kernel_float32(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = cosf(input[idx]);
        }
    }
int main() {
    size_t num_elements = 100;
    Tensor t;
    t.size = num_elements;
    t.ndim = 1;
    int shape[1] = { (int)num_elements };
    t.shape = shape;
    t.dtype = DTYPE_FLOAT32;
    float* h_data = (float*)malloc(num_elements * sizeof(float));
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = (float)i * 0.1f; // 0, 0.1, 0.2...
        }
    t.data = h_data;
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, num_elements * sizeof(float));
    cudaMalloc((void**)&d_output, num_elements * sizeof(float));
    cudaMemcpy(d_input, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    cosine_kernel_float32<<<blocks, threads>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();
    float* h_result = (float*)malloc(num_elements * sizeof(float));
    cudaMemcpy(h_result, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    printf("前10个cos值:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f", h_result[i]);
        }
    free(h_data);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
    }
