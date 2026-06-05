/**
  ******************************************************************************
  * @file        verify_tensor_scatter.cu
  * @author      Egor Izmaylov
  * @brief       提供 TensorScatter 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16

// 将 past_cache 复制到输出，为后续 cache 更新提供 functional scatter 的初始状态。
__global__ void copy_cache_kernel(const float* input, float* output, size_t total) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < total) {
        output[t] = input[t];
    }
}

// 按 update 的完整坐标替换 sequence 轴坐标，将新 cache 片段写入输出。
__global__ void tensor_scatter_kernel(
    const float* update,
    const long long* write_indices,
    float* output,
    const int* cache_shape,
    const int* update_shape,
    int rank,
    int axis,
    int mode,
    size_t update_len
) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= update_len || rank > MAX_RANK) return;

    int update_coords[MAX_RANK] = {0};
    int target_coords[MAX_RANK] = {0};
    size_t tmp = t;
    for (int d = rank - 1; d >= 0; --d) {
        update_coords[d] = (int)(tmp % (size_t)update_shape[d]);
        tmp /= (size_t)update_shape[d];
        target_coords[d] = update_coords[d];
    }

    int max_sequence_length = cache_shape[axis];
    long long write_start = write_indices ? write_indices[update_coords[0]] : 0;
    long long target_sequence = write_start + update_coords[axis];
    if (mode == 1) {
        target_sequence %= max_sequence_length;
        if (target_sequence < 0) target_sequence += max_sequence_length;
    } else if (target_sequence < 0 || target_sequence >= max_sequence_length) {
        return;
    }
    target_coords[axis] = (int)target_sequence;

    size_t output_index = 0;
    for (int d = 0; d < rank; ++d) {
        output_index = output_index * (size_t)cache_shape[d] + (size_t)target_coords[d];
    }
    output[output_index] = update[t];
}

// 计算形状数组对应的元素数量。
static size_t shape_size(const int* shape, int rank) {
    size_t size = 1;
    for (int d = 0; d < rank; ++d) {
        size *= (size_t)shape[d];
    }
    return size;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <past_cache.bin> <update.bin> <write_indices.bin|null> <params.bin> <out.bin>
    if (argc != 7) {
        printf("Usage: %s <out_len> <past_cache.bin> <update.bin> <write_indices.bin|null> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* cache_path = argv[2];
    const char* update_path = argv[3];
    const char* indices_path = argv[4];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) { printf("open params failed\n"); return 1; }
    int header[3];
    if (fread(header, sizeof(int), 3, fp) != 3) { printf("read params header failed\n"); fclose(fp); return 1; }
    int rank = header[0];
    int axis = header[1];
    int mode = header[2];
    if (rank <= 0 || rank > MAX_RANK) { printf("invalid rank\n"); fclose(fp); return 1; }
    if (axis < 0) axis += rank;
    if (axis <= 0 || axis >= rank) { printf("invalid axis\n"); fclose(fp); return 1; }

    int cache_shape[MAX_RANK] = {0};
    int update_shape[MAX_RANK] = {0};
    if (fread(cache_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        printf("read cache shape failed\n"); fclose(fp); return 1;
    }
    if (fread(update_shape, sizeof(int), (size_t)rank, fp) != (size_t)rank) {
        printf("read update shape failed\n"); fclose(fp); return 1;
    }
    fclose(fp);

    size_t cache_len = shape_size(cache_shape, rank);
    size_t update_len = shape_size(update_shape, rank);
    if (out_len != cache_len) { printf("out_len mismatch\n"); return 1; }

    size_t cache_bytes = cache_len * sizeof(float);
    size_t update_bytes = update_len * sizeof(float);
    size_t indices_bytes = (size_t)cache_shape[0] * sizeof(long long);

    float* h_cache = (float*)malloc(cache_bytes);
    float* h_update = (float*)malloc(update_bytes);
    float* h_out = (float*)malloc(cache_bytes);
    long long* h_indices = NULL;
    if (!h_cache || !h_update || !h_out) { printf("malloc failed\n"); return 1; }

    FILE* fc = fopen(cache_path, "rb");
    FILE* fu = fopen(update_path, "rb");
    if (!fc || !fu) { printf("open input failed\n"); return 1; }
    size_t rc = fread(h_cache, sizeof(float), cache_len, fc);
    size_t ru = fread(h_update, sizeof(float), update_len, fu);
    fclose(fc); fclose(fu);
    if (rc != cache_len || ru != update_len) { printf("fread mismatch\n"); return 1; }

    if (indices_path[0] != 'n') {
        h_indices = (long long*)malloc(indices_bytes);
        if (!h_indices) { printf("indices malloc failed\n"); return 1; }
        FILE* fi = fopen(indices_path, "rb");
        if (!fi) { printf("open indices failed\n"); return 1; }
        size_t ri = fread(h_indices, sizeof(long long), (size_t)cache_shape[0], fi);
        fclose(fi);
        if (ri != (size_t)cache_shape[0]) { printf("indices fread mismatch\n"); return 1; }
    }

    float *d_cache = NULL, *d_update = NULL, *d_out = NULL;
    long long* d_indices = NULL;
    int *d_cache_shape = NULL, *d_update_shape = NULL;
    cudaMalloc(&d_cache, cache_bytes);
    cudaMalloc(&d_update, update_bytes);
    cudaMalloc(&d_out, cache_bytes);
    cudaMalloc(&d_cache_shape, (size_t)rank * sizeof(int));
    cudaMalloc(&d_update_shape, (size_t)rank * sizeof(int));
    if (h_indices) cudaMalloc(&d_indices, indices_bytes);

    cudaMemcpy(d_cache, h_cache, cache_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_update, h_update, update_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cache_shape, cache_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_update_shape, update_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    if (h_indices) cudaMemcpy(d_indices, h_indices, indices_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int copy_blocks = (int)((cache_len + (size_t)threads - 1) / (size_t)threads);
    int update_blocks = (int)((update_len + (size_t)threads - 1) / (size_t)threads);
    copy_cache_kernel<<<copy_blocks, threads>>>(d_cache, d_out, cache_len);
    tensor_scatter_kernel<<<update_blocks, threads>>>(d_update, d_indices, d_out, d_cache_shape, d_update_shape, rank, axis, mode, update_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, cache_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t wo = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_cache); cudaFree(d_update); cudaFree(d_out);
    cudaFree(d_cache_shape); cudaFree(d_update_shape);
    if (d_indices) cudaFree(d_indices);
    free(h_cache); free(h_update); free(h_out);
    if (h_indices) free(h_indices);
    return 0;
}
