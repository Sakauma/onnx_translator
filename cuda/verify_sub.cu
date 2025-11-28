#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 核心核函数：只处理 float32，保证最高精度真值
__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

int main(int argc, char** argv) {
    // 统一接口：<数量> <输入A> <输入B> <输出>
    if (argc != 5) return 1; 
    size_t n = atol(argv[1]);
    size_t bytes = n * sizeof(float);
    
    // 1. 读取数据 (假设 Python 已经转好了 float32)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    
    FILE *fa = fopen(argv[2], "rb"); fread(h_a, 1, bytes, fa); fclose(fa);
    FILE *fb = fopen(argv[3], "rb"); fread(h_b, 1, bytes, fb); fclose(fb);
    
    // 2. GPU 计算
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    add_kernel<<<(n + 255)/256, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    
    // 3. 写入结果
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[4], "wb"); fwrite(h_out, 1, bytes, fout); fclose(fout);
    
    // 清理
    free(h_a); free(h_b); free(h_out);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
}