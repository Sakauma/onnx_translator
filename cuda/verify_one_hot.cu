/**
  ******************************************************************************
  * @file        verify_one_hot.cu
  * @author      Egor Izmaylov
  * @brief       提供 OneHot 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

struct OneHotParams {
    int32_t indices_rank;
    int32_t output_rank;
    int32_t axis;
    int32_t depth;
    int32_t indices_shape[MAX_RANK];
    int32_t output_shape[MAX_RANK + 1];
};

__global__ void one_hot_kernel(const int64_t* indices, const float* values, float* output, OneHotParams p, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    size_t tmp = tid;
    int32_t out_coords[MAX_RANK + 1] = {0};
    for (int d = p.output_rank - 1; d >= 0; --d) {
        int32_t dim = p.output_shape[d];
        out_coords[d] = dim > 0 ? (int32_t)(tmp % (size_t)dim) : 0;
        if (dim > 0) {
            tmp /= (size_t)dim;
        }
    }

    size_t indices_offset = 0;
    size_t stride = 1;
    for (int d = p.indices_rank - 1; d >= 0; --d) {
        int out_axis = d < p.axis ? d : d + 1;
        indices_offset += (size_t)out_coords[out_axis] * stride;
        stride *= (size_t)p.indices_shape[d];
    }

    int64_t target = indices[indices_offset];
    if (target < 0) {
        target += p.depth;
    }
    int32_t current = out_coords[p.axis];
    output[tid] = (target == current) ? values[1] : values[0];
}

static int read_params(const char* path, OneHotParams* p) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t header[4] = {0, 0, 0, 0};
    if (fread(header, sizeof(int32_t), 4, fp) != 4) {
        fclose(fp);
        fprintf(stderr, "read header failed\n");
        return 0;
    }
    p->indices_rank = header[0];
    p->output_rank = header[1];
    p->axis = header[2];
    p->depth = header[3];
    if (p->indices_rank < 0 || p->indices_rank > MAX_RANK || p->output_rank != p->indices_rank + 1) {
        fclose(fp);
        fprintf(stderr, "invalid rank\n");
        return 0;
    }
    if (fread(p->indices_shape, sizeof(int32_t), (size_t)p->indices_rank, fp) != (size_t)p->indices_rank) {
        fclose(fp);
        fprintf(stderr, "read indices shape failed\n");
        return 0;
    }
    if (fread(p->output_shape, sizeof(int32_t), (size_t)p->output_rank, fp) != (size_t)p->output_rank) {
        fclose(fp);
        fprintf(stderr, "read output shape failed\n");
        return 0;
    }
    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    // <out_len> <indices.bin> <depth.bin> <values.bin> <params.bin> <out.bin>
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <indices.bin> <depth.bin> <values.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* indices_path = argv[2];
    const char* values_path = argv[4];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    OneHotParams params;
    if (!read_params(params_path, &params)) {
        return 1;
    }

    size_t indices_len = 1;
    for (int d = 0; d < params.indices_rank; ++d) {
        indices_len *= (size_t)params.indices_shape[d];
    }

    int64_t* h_indices = (int64_t*)malloc(indices_len * sizeof(int64_t));
    float h_values[2] = {0.0f, 1.0f};
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_indices || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(indices_path, "rb");
    if (!fi || fread(h_indices, sizeof(int64_t), indices_len, fi) != indices_len) {
        if (fi) fclose(fi);
        fprintf(stderr, "read indices failed\n");
        return 1;
    }
    fclose(fi);

    FILE* fv = fopen(values_path, "rb");
    if (!fv || fread(h_values, sizeof(float), 2, fv) != 2) {
        if (fv) fclose(fv);
        fprintf(stderr, "read values failed\n");
        return 1;
    }
    fclose(fv);

    int64_t* d_indices = NULL;
    float* d_values = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_indices, indices_len * sizeof(int64_t));
    cudaMalloc(&d_values, 2 * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMemcpy(d_indices, h_indices, indices_len * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, 2 * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    one_hot_kernel<<<blocks, threads>>>(d_indices, d_values, d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo || fwrite(h_output, sizeof(float), out_len, fo) != out_len) {
        if (fo) fclose(fo);
        fprintf(stderr, "write output failed\n");
        return 1;
    }
    fclose(fo);

    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_output);
    free(h_indices);
    free(h_output);
    return 0;
}
