/**
  ******************************************************************************
  * @file        verify_scatter_elements.cu
  * @author      Egor Izmaylov
  * @brief       提供 ScatterElements 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 8

// 保存 ScatterElements 参考计算所需的 rank、axis、reduction 以及 data/updates 形状。
struct ScatterElementsParams {
    int32_t rank;
    int32_t axis;
    int32_t reduction;
    int32_t data_shape[MAX_RANK];
    int32_t updates_shape[MAX_RANK];
};

// 按 updates 坐标与 indices 指定的目标 axis 坐标，写入或归约到输出张量。
__global__ void scatter_elements_kernel(
    float* output,
    const long long* indices,
    const float* updates,
    ScatterElementsParams p,
    size_t updates_len
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= updates_len) return;

    size_t tmp = tid;
    int32_t coords[MAX_RANK];
    for (int d = p.rank - 1; d >= 0; --d) {
        int32_t dim = p.updates_shape[d];
        coords[d] = dim > 0 ? (int32_t)(tmp % (size_t)dim) : 0;
        if (dim > 0) {
            tmp /= (size_t)dim;
        }
    }

    long long axis_index = indices[tid];
    int32_t axis_dim = p.data_shape[p.axis];
    if (axis_index < 0) {
        axis_index += axis_dim;
    }
    if (axis_index < 0) {
        axis_index = 0;
    }
    if (axis_index >= axis_dim) {
        axis_index = axis_dim - 1;
    }
    coords[p.axis] = (int32_t)axis_index;

    size_t output_index = 0;
    size_t stride = 1;
    for (int d = p.rank - 1; d >= 0; --d) {
        output_index += (size_t)coords[d] * stride;
        stride *= (size_t)p.data_shape[d];
    }

    if (p.reduction == 1) {
        output[output_index] += updates[tid];
    } else if (p.reduction == 2) {
        output[output_index] *= updates[tid];
    } else {
        output[output_index] = updates[tid];
    }
}

// 顺序读取 `[rank, axis, reduction, data_shape..., updates_shape...]` 参数。
static int read_scatter_elements_params(const char* params_path, ScatterElementsParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t header[3];
    if (fread(header, sizeof(int32_t), 3, fp) != 3) {
        fclose(fp);
        fprintf(stderr, "read header failed\n");
        return 0;
    }
    params->rank = header[0];
    params->axis = header[1];
    params->reduction = header[2];
    if (params->rank <= 0 || params->rank > MAX_RANK || params->axis < 0 || params->axis >= params->rank) {
        fclose(fp);
        fprintf(stderr, "invalid params\n");
        return 0;
    }
    if (fread(params->data_shape, sizeof(int32_t), (size_t)params->rank, fp) != (size_t)params->rank) {
        fclose(fp);
        fprintf(stderr, "read data shape failed\n");
        return 0;
    }
    if (fread(params->updates_shape, sizeof(int32_t), (size_t)params->rank, fp) != (size_t)params->rank) {
        fclose(fp);
        fprintf(stderr, "read updates shape failed\n");
        return 0;
    }
    fclose(fp);
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <data.bin> <indices.bin> <updates.bin> <params.bin> <out.bin>
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <indices.bin> <updates.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* indices_path = argv[3];
    const char* updates_path = argv[4];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    ScatterElementsParams params;
    if (!read_scatter_elements_params(params_path, &params)) {
        return 1;
    }

    size_t data_len = 1;
    size_t updates_len = 1;
    for (int d = 0; d < params.rank; ++d) {
        data_len *= (size_t)params.data_shape[d];
        updates_len *= (size_t)params.updates_shape[d];
    }
    if (out_len != data_len) {
        fprintf(stderr, "out_len mismatch: got %zu expected %zu\n", out_len, data_len);
        return 1;
    }

    float* h_data = (float*)malloc(data_len * sizeof(float));
    long long* h_indices = (long long*)malloc(updates_len * sizeof(long long));
    float* h_updates = (float*)malloc(updates_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_data || !h_indices || !h_updates || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* data_fp = fopen(data_path, "rb");
    FILE* indices_fp = fopen(indices_path, "rb");
    FILE* updates_fp = fopen(updates_path, "rb");
    if (!data_fp || !indices_fp || !updates_fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t data_read = fread(h_data, sizeof(float), data_len, data_fp);
    size_t indices_read = fread(h_indices, sizeof(long long), updates_len, indices_fp);
    size_t updates_read = fread(h_updates, sizeof(float), updates_len, updates_fp);
    fclose(data_fp);
    fclose(indices_fp);
    fclose(updates_fp);
    if (data_read != data_len || indices_read != updates_len || updates_read != updates_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_output = NULL;
    long long* d_indices = NULL;
    float* d_updates = NULL;
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMalloc(&d_indices, updates_len * sizeof(long long));
    cudaMalloc(&d_updates, updates_len * sizeof(float));
    cudaMemcpy(d_output, h_data, out_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, updates_len * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_updates, h_updates, updates_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((updates_len + (size_t)threads - 1) / (size_t)threads);
    scatter_elements_kernel<<<blocks, threads>>>(d_output, d_indices, d_updates, params, updates_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t written = fwrite(h_output, sizeof(float), out_len, out_fp);
    fclose(out_fp);
    if (written != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_updates);
    free(h_data);
    free(h_indices);
    free(h_updates);
    free(h_output);
    return 0;
}
