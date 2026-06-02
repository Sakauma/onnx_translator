/*
 * 文件功能：提供 relu 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 实现 `relu_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void relu_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
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