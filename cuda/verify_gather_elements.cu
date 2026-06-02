/*
 * 文件功能：提供 gather elements 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct GatherElementsParams {
    int M;
    int N;
    int axis; 
};

// 实现 `gatherelements_axis1_2d` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__global__ void gatherelements_axis1_2d(
    const float* data, const long long* idx, float* out, int M, int N
) {
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = M * N;
    if (t < total) {
        int i = t / N;
        // int j = t % N;
        long long col = idx[t];
        if (col < 0) col += N;
        out[t] = data[i * N + (int)col];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* idx_path  = argv[3];
    const char* p_path    = argv[4];
    const char* out_path  = argv[5];

    GatherElementsParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) { fprintf(stderr, "open params failed\n"); return 1; }
    size_t pr = fread(&p, sizeof(GatherElementsParams), 1, fp);
    fclose(fp);
    if (pr != 1) { fprintf(stderr, "read params failed\n"); return 1; }
    if (p.axis != 1) { fprintf(stderr, "This verifier only supports axis=1 for 2D.\n"); return 1; }

    int M = p.M, N = p.N;
    if (out_len != (size_t)(M * N)) {
        fprintf(stderr, "out_len mismatch: got %zu expected %d\n", out_len, M*N);
        return 1;
    }

    size_t bytes = out_len * sizeof(float);
    size_t idx_bytes = out_len * sizeof(long long);

    float* h_data = (float*)malloc(bytes);
    long long* h_idx = (long long*)malloc(idx_bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_data || !h_idx || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE* fd = fopen(data_path, "rb");
    FILE* fi = fopen(idx_path, "rb");
    if (!fd || !fi) { fprintf(stderr, "open input failed\n"); return 1; }

    size_t rd = fread(h_data, sizeof(float), out_len, fd);
    size_t ri = fread(h_idx, sizeof(long long), out_len, fi);
    fclose(fd); fclose(fi);
    if (rd != out_len || ri != out_len) { fprintf(stderr, "fread mismatch\n"); return 1; }

    float *d_data=NULL, *d_out=NULL;
    long long* d_idx=NULL;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_idx, idx_bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, idx_bytes, cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(int)((out_len + threads -1)/threads);
    gatherelements_axis1_2d<<<blocks,threads>>>(d_data, d_idx, d_out, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { fprintf(stderr, "open output failed\n"); return 1; }
    size_t w = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (w != out_len) { fprintf(stderr, "fwrite mismatch\n"); return 1; }

    cudaFree(d_data); cudaFree(d_idx); cudaFree(d_out);
    free(h_data); free(h_idx); free(h_out);
    return 0;
}
