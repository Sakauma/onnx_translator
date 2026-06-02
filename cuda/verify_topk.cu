#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <cuda_runtime.h>

struct TopKParams {
    int32_t M;
    int32_t N;
    int32_t axis;
    int32_t k;
    int32_t largest;
    int32_t sorted_flag;
};

// Egor Izmaylov: Function `topk_axis1_2d_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void topk_axis1_2d_kernel(
    const float* data,
    float* out_vals,
    int64_t* out_idx,
    int M,
    int N,
    int K
) {
    int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= M) return;

    const float* row_ptr = data + (size_t)row * N;
    float* vals_ptr = out_vals + (size_t)row * K;
    int64_t* idx_ptr = out_idx + (size_t)row * K;

    // 固定主路径：largest=1, sorted=1
    // 简单 verifier：每次选一个最大值
    // 为了简单限制 N 不要太大
    bool used[1024];
    if (N > 1024) return;

    for (int j = 0; j < N; ++j) used[j] = false;

    for (int k = 0; k < K; ++k) {
        float best_v = -FLT_MAX;
        int best_i = 0;
        bool found = false;

        for (int j = 0; j < N; ++j) {
            if (used[j]) continue;
            float v = row_ptr[j];
            // tie 时选较小下标
            if (!found || v > best_v || (v == best_v && j < best_i)) {
                best_v = v;
                best_i = j;
                found = true;
            }
        }

        used[best_i] = true;
        vals_ptr[k] = best_v;
        idx_ptr[k] = (int64_t)best_i;
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <data.bin> <k.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <k.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* k_path = argv[3];
    const char* p_path = argv[4];
    const char* out_path = argv[5];

    TopKParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(TopKParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (p.axis != 1 || p.largest != 1 || p.sorted_flag != 1) {
        fprintf(stderr, "This verifier only supports axis=1, largest=1, sorted=1.\n");
        return 1;
    }

    if ((size_t)(p.M * p.k) != out_len) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    if (p.N > 1024) {
        fprintf(stderr, "N too large for this simple verifier\n");
        return 1;
    }

    int64_t h_k_input = 0;
    FILE* fk = fopen(k_path, "rb");
    if (!fk) {
        fprintf(stderr, "open k input failed\n");
        return 1;
    }
    if (fread(&h_k_input, sizeof(int64_t), 1, fk) != 1) {
        fprintf(stderr, "read k input failed\n");
        fclose(fk);
        return 1;
    }
    fclose(fk);

    if ((int)h_k_input != p.k) {
        fprintf(stderr, "k mismatch between input and params\n");
        return 1;
    }

    size_t data_len = (size_t)p.M * p.N;
    float* h_data = (float*)malloc(data_len * sizeof(float));
    float* h_vals = (float*)malloc(out_len * sizeof(float));
    int64_t* h_idx = (int64_t*)malloc(out_len * sizeof(int64_t));
    if (!h_data || !h_vals || !h_idx) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fd = fopen(data_path, "rb");
    if (!fd) {
        fprintf(stderr, "open data failed\n");
        return 1;
    }
    if (fread(h_data, sizeof(float), data_len, fd) != data_len) {
        fprintf(stderr, "read data failed\n");
        fclose(fd);
        return 1;
    }
    fclose(fd);

    float* d_data = NULL;
    float* d_vals = NULL;
    int64_t* d_idx = NULL;

    cudaMalloc(&d_data, data_len * sizeof(float));
    cudaMalloc(&d_vals, out_len * sizeof(float));
    cudaMalloc(&d_idx, out_len * sizeof(int64_t));

    cudaMemcpy(d_data, h_data, data_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (p.M + threads - 1) / threads;
    topk_axis1_2d_kernel<<<blocks, threads>>>(d_data, d_vals, d_idx, p.M, p.N, p.k);
    cudaDeviceSynchronize();

    cudaMemcpy(h_vals, d_vals, out_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx, d_idx, out_len * sizeof(int64_t), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open values out failed\n");
        return 1;
    }
    if (fwrite(h_vals, sizeof(float), out_len, fo) != out_len) {
        fprintf(stderr, "write values failed\n");
        fclose(fo);
        return 1;
    }
    fclose(fo);

    FILE* fi = fopen("tmp_out_idx.bin", "wb");
    if (!fi) {
        fprintf(stderr, "open tmp_out_idx.bin failed\n");
        return 1;
    }
    if (fwrite(h_idx, sizeof(int64_t), out_len, fi) != out_len) {
        fprintf(stderr, "write indices failed\n");
        fclose(fi);
        return 1;
    }
    fclose(fi);

    cudaFree(d_data);
    cudaFree(d_vals);
    cudaFree(d_idx);
    free(h_data);
    free(h_vals);
    free(h_idx);
    return 0;
}