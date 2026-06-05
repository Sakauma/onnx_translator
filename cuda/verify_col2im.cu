/**
  ******************************************************************************
  * @file        verify_col2im.cu
  * @author      Egor Izmaylov
  * @brief       提供 Col2Im 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define COL2IM_MAX_RANK 8

struct Col2ImParams {
    int32_t batch;
    int32_t channels;
    int32_t spatial_rank;
    int32_t columns;
    int32_t kernel_size;
    int32_t block_count;
    int32_t image_dims[COL2IM_MAX_RANK];
    int32_t block_dims[COL2IM_MAX_RANK];
    int32_t pads[COL2IM_MAX_RANK * 2];
    int32_t strides[COL2IM_MAX_RANK];
    int32_t dilations[COL2IM_MAX_RANK];
    int32_t n_blocks[COL2IM_MAX_RANK];
};

// 将线性下标按指定形状展开为坐标，CUDA reference 使用固定上限避免动态分配。
__device__ void col2im_unravel(size_t index, const int32_t* shape, int rank, int32_t* coords) {
    for (int axis = rank - 1; axis >= 0; --axis) {
        coords[axis] = (int32_t)(index % (size_t)shape[axis]);
        index /= (size_t)shape[axis];
    }
}

// 实现 Col2Im CUDA reference kernel，每个输出元素反查所有列块贡献并完成重叠累加。
__global__ void col2im_kernel(const double* input, double* output, Col2ImParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    size_t tmp = tid;
    int32_t out_coords[COL2IM_MAX_RANK] = {0};
    for (int axis = p.spatial_rank - 1; axis >= 0; --axis) {
        out_coords[axis] = (int32_t)(tmp % (size_t)p.image_dims[axis]);
        tmp /= (size_t)p.image_dims[axis];
    }
    int32_t c = (int32_t)(tmp % (size_t)p.channels);
    tmp /= (size_t)p.channels;
    int32_t n = (int32_t)tmp;

    double sum = 0.0;
    int32_t kernel_coords[COL2IM_MAX_RANK] = {0};
    int32_t block_coords[COL2IM_MAX_RANK] = {0};
    for (int32_t k = 0; k < p.kernel_size; ++k) {
        col2im_unravel((size_t)k, p.block_dims, p.spatial_rank, kernel_coords);
        for (int32_t col = 0; col < p.block_count; ++col) {
            col2im_unravel((size_t)col, p.n_blocks, p.spatial_rank, block_coords);
            bool matches = true;
            for (int axis = 0; axis < p.spatial_rank; ++axis) {
                int32_t image_coord = block_coords[axis] * p.strides[axis]
                                    - p.pads[axis]
                                    + kernel_coords[axis] * p.dilations[axis];
                if (image_coord != out_coords[axis]) {
                    matches = false;
                    break;
                }
            }
            if (!matches) continue;

            size_t input_index = ((size_t)n * (size_t)(p.channels * p.kernel_size)
                               + (size_t)c * (size_t)p.kernel_size
                               + (size_t)k) * (size_t)p.block_count
                               + (size_t)col;
            sum += input[input_index];
        }
    }

    output[tid] = sum;
}

// 读取二进制文件到 vector，供 main 函数加载输入和参数。
template <typename T>
static int read_vector(const char* path, std::vector<T>& data) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t count = fread(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
    return count == data.size();
}

// 将 Python runner 打包的 int32 参数解码为固定结构，避免 CUDA kernel 依赖可变长度数组。
static int parse_params(const char* path, Col2ImParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    rewind(fp);
    if (bytes <= 0 || bytes % (long)sizeof(int32_t) != 0) {
        fclose(fp);
        return 0;
    }
    size_t count = (size_t)bytes / sizeof(int32_t);
    std::vector<int32_t> raw(count);
    if (fread(raw.data(), sizeof(int32_t), count, fp) != count) {
        fclose(fp);
        return 0;
    }
    fclose(fp);

    if (count < 6) return 0;
    params->batch = raw[0];
    params->channels = raw[1];
    params->spatial_rank = raw[2];
    params->columns = raw[3];
    params->kernel_size = raw[4];
    params->block_count = raw[5];
    int rank = params->spatial_rank;
    if (rank < 2 || rank > COL2IM_MAX_RANK) return 0;
    size_t expected = (size_t)6 + (size_t)rank + (size_t)rank + (size_t)(2 * rank) + (size_t)rank * 3;
    if (count != expected) return 0;

    size_t offset = 6;
    for (int i = 0; i < rank; ++i) params->image_dims[i] = raw[offset++];
    for (int i = 0; i < rank; ++i) params->block_dims[i] = raw[offset++];
    for (int i = 0; i < 2 * rank; ++i) params->pads[i] = raw[offset++];
    for (int i = 0; i < rank; ++i) params->strides[i] = raw[offset++];
    for (int i = 0; i < rank; ++i) params->dilations[i] = raw[offset++];
    for (int i = 0; i < rank; ++i) params->n_blocks[i] = raw[offset++];
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <image_shape.bin> <block_shape.bin> <params.bin> <out.bin>
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <image_shape.bin> <block_shape.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[5];
    const char* out_path = argv[6];

    Col2ImParams params = {};
    if (!parse_params(params_path, &params)) {
        fprintf(stderr, "invalid params\n");
        return 2;
    }

    size_t expected_input_len = (size_t)params.batch
                              * (size_t)params.channels
                              * (size_t)params.kernel_size
                              * (size_t)params.block_count;
    size_t expected_output_len = (size_t)params.batch * (size_t)params.channels;
    for (int axis = 0; axis < params.spatial_rank; ++axis) {
        expected_output_len *= (size_t)params.image_dims[axis];
    }
    if (out_len != expected_output_len || (size_t)params.columns != (size_t)params.block_count) {
        fprintf(stderr, "shape mismatch\n");
        return 3;
    }

    std::vector<double> h_input(expected_input_len);
    std::vector<double> h_output(out_len);
    if (!read_vector(input_path, h_input)) {
        fprintf(stderr, "read input failed\n");
        return 4;
    }

    double* d_input = NULL;
    double* d_output = NULL;
    cudaMalloc((void**)&d_input, expected_input_len * sizeof(double));
    cudaMalloc((void**)&d_output, out_len * sizeof(double));
    cudaMemcpy(d_input, h_input.data(), expected_input_len * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    col2im_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, out_len * sizeof(double), cudaMemcpyDeviceToHost);

    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        return 5;
    }
    if (fwrite(h_output.data(), sizeof(double), out_len, fp) != out_len) {
        fprintf(stderr, "write output failed\n");
        fclose(fp);
        return 6;
    }
    fclose(fp);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
