/**
  ******************************************************************************
  * @file        verify_triangular_common.cuh
  * @author      Egor Izmaylov
  * @brief       保存三角矩阵类 CUDA verifier 共用的参考实现。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 8

struct TriangularParams {
    int32_t rank;
    int32_t upper;
    int32_t k;
    int32_t dims[MAX_RANK];
};

__global__ void triangular_kernel(const float* input, float* output, TriangularParams params, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    size_t tmp = tid;
    int32_t col = 0;
    int32_t row = 0;
    for (int d = params.rank - 1; d >= 0; --d) {
        int32_t dim = params.dims[d];
        int32_t coord = dim > 0 ? (int32_t)(tmp % (size_t)dim) : 0;
        if (dim > 0) {
            tmp /= (size_t)dim;
        }
        if (d == params.rank - 1) {
            col = coord;
        } else if (d == params.rank - 2) {
            row = coord;
        }
    }

    int keep = params.upper ? (col - row >= params.k) : (row - col >= -params.k);
    output[tid] = keep ? input[tid] : 0.0f;
}

static int read_triangular_params(const char* path, TriangularParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t header[3] = {0, 0, 0};
    if (fread(header, sizeof(int32_t), 3, fp) != 3) {
        fclose(fp);
        fprintf(stderr, "read params header failed\n");
        return 0;
    }
    params->rank = header[0];
    params->upper = header[1];
    params->k = header[2];
    if (params->rank <= 0 || params->rank > MAX_RANK) {
        fclose(fp);
        fprintf(stderr, "invalid rank\n");
        return 0;
    }
    if (fread(params->dims, sizeof(int32_t), (size_t)params->rank, fp) != (size_t)params->rank) {
        fclose(fp);
        fprintf(stderr, "read dims failed\n");
        return 0;
    }
    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    // <out_len> <data.bin> <k.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <k.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    TriangularParams params;
    if (!read_triangular_params(params_path, &params)) {
        return 1;
    }

    size_t expected = 1;
    for (int i = 0; i < params.rank; ++i) {
        expected *= (size_t)params.dims[i];
    }
    if (out_len != expected) {
        fprintf(stderr, "out_len mismatch: got %zu expected %zu\n", out_len, expected);
        return 1;
    }

    float* h_input = (float*)malloc(out_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(data_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t read_count = fread(h_input, sizeof(float), out_len, fi);
    fclose(fi);
    if (read_count != out_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, out_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMemcpy(d_input, h_input, out_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    triangular_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t write_count = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
