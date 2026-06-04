/**
  ******************************************************************************
  * @file        verify_concat.cu
  * @author      Egor Izmaylov
  * @brief       提供 concat 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 16
#define MAX_INPUTS 16

// 根据输出坐标在 concat axis 上的段落，定位来源输入和该输入内的局部坐标。
__global__ void concat_kernel(
    const float* packed_inputs,
    float* output,
    const int* shapes,
    const int* output_shape,
    const size_t* input_offsets,
    const int* axis_offsets,
    int num_inputs,
    int rank,
    int axis,
    size_t out_len
) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t >= out_len || rank > MAX_RANK || num_inputs > MAX_INPUTS) return;

    int coords[MAX_RANK] = {0};
    size_t tmp = t;
    for (int d = rank - 1; d >= 0; --d) {
        coords[d] = (int)(tmp % (size_t)output_shape[d]);
        tmp /= (size_t)output_shape[d];
    }

    int source = num_inputs - 1;
    for (int i = 0; i < num_inputs; ++i) {
        int begin = axis_offsets[i];
        int end = axis_offsets[i + 1];
        if (coords[axis] >= begin && coords[axis] < end) {
            source = i;
            coords[axis] -= begin;
            break;
        }
    }

    size_t local_idx = 0;
    for (int d = 0; d < rank; ++d) {
        local_idx = local_idx * (size_t)shapes[source * rank + d] + (size_t)coords[d];
    }
    output[t] = packed_inputs[input_offsets[source] + local_idx];
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
    // <out_len> <input0.bin> ... <inputN.bin> <params.bin> <out.bin>
    if (argc < 5) {
        printf("Usage: %s <out_len> <input*.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[argc - 2];
    const char* out_path = argv[argc - 1];

    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        printf("open params failed\n");
        return 1;
    }

    int header[3] = {0};
    if (fread(header, sizeof(int), 3, fp) != 3) {
        fclose(fp);
        printf("read header failed\n");
        return 1;
    }
    int num_inputs = header[0];
    int rank = header[1];
    int axis = header[2];
    if (num_inputs <= 0 || num_inputs > MAX_INPUTS || rank <= 0 || rank > MAX_RANK || axis < 0 || axis >= rank) {
        fclose(fp);
        printf("invalid concat params\n");
        return 1;
    }
    if (argc != num_inputs + 4) {
        fclose(fp);
        printf("argc/input count mismatch\n");
        return 1;
    }

    int shapes[MAX_INPUTS * MAX_RANK] = {0};
    if (fread(shapes, sizeof(int), (size_t)num_inputs * (size_t)rank, fp) != (size_t)num_inputs * (size_t)rank) {
        fclose(fp);
        printf("read shapes failed\n");
        return 1;
    }
    fclose(fp);

    int output_shape[MAX_RANK] = {0};
    for (int d = 0; d < rank; ++d) {
        output_shape[d] = shapes[d];
    }
    output_shape[axis] = 0;

    size_t lengths[MAX_INPUTS] = {0};
    size_t offsets[MAX_INPUTS] = {0};
    int axis_offsets[MAX_INPUTS + 1] = {0};
    size_t total_input_len = 0;
    for (int i = 0; i < num_inputs; ++i) {
        const int* shape = &shapes[i * rank];
        for (int d = 0; d < rank; ++d) {
            if (shape[d] <= 0 || (d != axis && shape[d] != output_shape[d])) {
                printf("invalid input shape\n");
                return 1;
            }
        }
        lengths[i] = shape_size(shape, rank);
        offsets[i] = total_input_len;
        total_input_len += lengths[i];
        axis_offsets[i + 1] = axis_offsets[i] + shape[axis];
        output_shape[axis] += shape[axis];
    }

    size_t expected_out_len = shape_size(output_shape, rank);
    if (out_len != expected_out_len) {
        printf("out_len mismatch: got %zu expected %zu\n", out_len, expected_out_len);
        return 1;
    }

    float* h_inputs = (float*)malloc(total_input_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_inputs || !h_output) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < num_inputs; ++i) {
        FILE* fi = fopen(argv[2 + i], "rb");
        if (!fi) {
            printf("open input failed\n");
            return 1;
        }
        size_t ri = fread(h_inputs + offsets[i], sizeof(float), lengths[i], fi);
        fclose(fi);
        if (ri != lengths[i]) {
            printf("fread mismatch\n");
            return 1;
        }
    }

    float* d_inputs = NULL;
    float* d_output = NULL;
    int* d_shapes = NULL;
    int* d_output_shape = NULL;
    size_t* d_offsets = NULL;
    int* d_axis_offsets = NULL;
    cudaMalloc(&d_inputs, total_input_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMalloc(&d_shapes, (size_t)num_inputs * (size_t)rank * sizeof(int));
    cudaMalloc(&d_output_shape, (size_t)rank * sizeof(int));
    cudaMalloc(&d_offsets, (size_t)num_inputs * sizeof(size_t));
    cudaMalloc(&d_axis_offsets, (size_t)(num_inputs + 1) * sizeof(int));

    cudaMemcpy(d_inputs, h_inputs, total_input_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shapes, shapes, (size_t)num_inputs * (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shape, output_shape, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, (size_t)num_inputs * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_axis_offsets, axis_offsets, (size_t)(num_inputs + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    concat_kernel<<<blocks, threads>>>(
        d_inputs,
        d_output,
        d_shapes,
        d_output_shape,
        d_offsets,
        d_axis_offsets,
        num_inputs,
        rank,
        axis,
        out_len
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        printf("open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) {
        printf("fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_inputs);
    cudaFree(d_output);
    cudaFree(d_shapes);
    cudaFree(d_output_shape);
    cudaFree(d_offsets);
    cudaFree(d_axis_offsets);
    free(h_inputs);
    free(h_output);
    return 0;
}
