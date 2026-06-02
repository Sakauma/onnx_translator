/**
  ******************************************************************************
  * @file        verify_gather.cu
  * @author      Egor Izmaylov
  * @brief       提供 gather 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct GatherParams {
    int M;
    int N;
    int I;
};

// 实现 `gather_axis0_2d_1d` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__global__ void gather_axis0_2d_1d(
    const float* data, const long long* idx, float* out, int M, int N, int I
) {
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = I * N;
    if (t < total) {
        int i = t / N;      // indices 的位置
        int j = t % N;      // 列
        long long r = idx[i];
        out[t] = data[(int)r * N + j];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>
    if (argc != 6) {
        printf("Usage: %s <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* idx_path  = argv[3];
    const char* p_path    = argv[4];
    const char* out_path  = argv[5];

    GatherParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) { printf("open params failed\n"); return 1; }
    size_t pr = fread(&p, sizeof(GatherParams), 1, fp);
    fclose(fp);
    if (pr != 1) { printf("read params failed\n"); return 1; }

    int M = p.M, N = p.N, I = p.I;
    if (out_len != (size_t)(I * N)) {
        printf("out_len mismatch: got %zu expected %d\n", out_len, I * N);
        return 1;
    }

    size_t data_len = (size_t)M * (size_t)N;
    size_t idx_len  = (size_t)I;
    size_t data_bytes = data_len * sizeof(float);
    size_t idx_bytes  = idx_len  * sizeof(long long);
    size_t out_bytes  = out_len  * sizeof(float);

    float* h_data = (float*)malloc(data_bytes);
    long long* h_idx = (long long*)malloc(idx_bytes);
    float* h_out = (float*)malloc(out_bytes);
    if (!h_data || !h_idx || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fd = fopen(data_path, "rb");
    FILE* fi = fopen(idx_path, "rb");
    if (!fd || !fi) { printf("open input failed\n"); return 1; }

    size_t rd = fread(h_data, sizeof(float), data_len, fd);
    size_t ri = fread(h_idx, sizeof(long long), idx_len, fi);
    fclose(fd); fclose(fi);
    if (rd != data_len || ri != idx_len) { printf("fread mismatch\n"); return 1; }

    float *d_data = NULL, *d_out = NULL;
    long long* d_idx = NULL;
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_idx,  idx_bytes);
    cudaMalloc(&d_out,  out_bytes);

    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,  h_idx,  idx_bytes,  cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    gather_axis0_2d_1d<<<blocks, threads>>>(d_data, d_idx, d_out, M, N, I);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t wo = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_data); cudaFree(d_idx); cudaFree(d_out);
    free(h_data); free(h_idx); free(h_out);
    return 0;
}
