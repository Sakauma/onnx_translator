/**
  ******************************************************************************
  * @file        verify_mean_variance_normalization.cu
  * @author      Egor Izmaylov
  * @brief       提供 MeanVarianceNormalization 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MVN_MAX_NDIM 8

// 保存 MeanVarianceNormalization 参考计算所需的 rank、shape 和 axes。
struct MvnParams {
    int32_t rank;
    int32_t num_axes;
    int32_t shape[MVN_MAX_NDIM];
    int32_t axes[MVN_MAX_NDIM];
};

// 将扁平索引还原为多维坐标。
__device__ void mvn_coords_from_index(size_t index, const int32_t* shape, int rank, int* coords) {
    for (int i = rank - 1; i >= 0; i--) {
        coords[i] = (int)(index % (size_t)shape[i]);
        index /= (size_t)shape[i];
    }
}

// 将多维坐标映射回扁平索引。
__device__ size_t mvn_index_from_coords(const int* coords, const int32_t* shape, int rank) {
    size_t index = 0;
    for (int i = 0; i < rank; i++) {
        index = index * (size_t)shape[i] + (size_t)coords[i];
    }
    return index;
}

// 按 ONNX MeanVarianceNormalization 公式计算 `(x - mean) / sqrt(variance)`。
__global__ void mvn_kernel(const double* input, double* output, MvnParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int base_coords[MVN_MAX_NDIM] = {0};
    mvn_coords_from_index(tid, p.shape, p.rank, base_coords);

    size_t reduce_steps = 1;
    for (int i = 0; i < p.num_axes; i++) {
        reduce_steps *= (size_t)p.shape[p.axes[i]];
    }

    double sum = 0.0;
    for (size_t r = 0; r < reduce_steps; r++) {
        int coords[MVN_MAX_NDIM];
        for (int i = 0; i < p.rank; i++) coords[i] = base_coords[i];

        size_t temp = r;
        for (int k = p.num_axes - 1; k >= 0; k--) {
            int axis = p.axes[k];
            coords[axis] = (int)(temp % (size_t)p.shape[axis]);
            temp /= (size_t)p.shape[axis];
        }
        sum += input[mvn_index_from_coords(coords, p.shape, p.rank)];
    }
    double mean = sum / (double)reduce_steps;

    double square_sum = 0.0;
    for (size_t r = 0; r < reduce_steps; r++) {
        int coords[MVN_MAX_NDIM];
        for (int i = 0; i < p.rank; i++) coords[i] = base_coords[i];

        size_t temp = r;
        for (int k = p.num_axes - 1; k >= 0; k--) {
            int axis = p.axes[k];
            coords[axis] = (int)(temp % (size_t)p.shape[axis]);
            temp /= (size_t)p.shape[axis];
        }
        double diff = input[mvn_index_from_coords(coords, p.shape, p.rank)] - mean;
        square_sum += diff * diff;
    }
    double variance = square_sum / (double)reduce_steps;
    output[tid] = (input[tid] - mean) / sqrt(variance);
}

// 读取变长 rank/axes 参数，避免结构体 padding 影响二进制兼容。
static int read_mvn_params(const char* params_path, MvnParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }

    int32_t header[2];
    if (fread(header, sizeof(int32_t), 2, fp) != 2) {
        fprintf(stderr, "read params header failed\n");
        fclose(fp);
        return 0;
    }
    params->rank = header[0];
    params->num_axes = header[1];
    if (params->rank <= 0 || params->rank > MVN_MAX_NDIM || params->num_axes <= 0 || params->num_axes > MVN_MAX_NDIM) {
        fprintf(stderr, "invalid params header\n");
        fclose(fp);
        return 0;
    }
    for (int i = 0; i < MVN_MAX_NDIM; i++) {
        params->shape[i] = 1;
        params->axes[i] = 0;
    }
    if (fread(params->shape, sizeof(int32_t), (size_t)params->rank, fp) != (size_t)params->rank) {
        fprintf(stderr, "read shape failed\n");
        fclose(fp);
        return 0;
    }
    if (fread(params->axes, sizeof(int32_t), (size_t)params->num_axes, fp) != (size_t)params->num_axes) {
        fprintf(stderr, "read axes failed\n");
        fclose(fp);
        return 0;
    }
    fclose(fp);

    for (int i = 0; i < params->rank; i++) {
        if (params->shape[i] <= 0) return 0;
    }
    for (int i = 0; i < params->num_axes; i++) {
        if (params->axes[i] < 0 || params->axes[i] >= params->rank) return 0;
    }
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    MvnParams params;
    if (!read_mvn_params(params_path, &params)) {
        return 1;
    }

    size_t expected_len = 1;
    for (int i = 0; i < params.rank; i++) {
        expected_len *= (size_t)params.shape[i];
    }
    if (out_len != expected_len) {
        fprintf(stderr, "output length mismatch\n");
        return 1;
    }

    size_t bytes = out_len * sizeof(double);
    double* h_input = (double*)malloc(bytes);
    double* h_output = (double*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }

    FILE* fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }
    size_t read_count = fread(h_input, sizeof(double), out_len, fp);
    fclose(fp);
    if (read_count != out_len) {
        fprintf(stderr, "read input failed\n");
        free(h_input);
        free(h_output);
        return 1;
    }

    double* d_input = NULL;
    double* d_output = NULL;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    mvn_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return 1;
    }
    size_t write_count = fwrite(h_output, sizeof(double), out_len, fp);
    fclose(fp);
    if (write_count != out_len) {
        fprintf(stderr, "write output failed\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
