/**
  ******************************************************************************
  * @file        verify_depth_to_space.cu
  * @author      Egor Izmaylov
  * @brief       提供 DepthToSpace 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 保存 DepthToSpace 参考计算所需的 NCHW 输入形状、blocksize 和 mode。
struct DepthToSpaceParams {
    int32_t batch;
    int32_t channels;
    int32_t height;
    int32_t width;
    int32_t blocksize;
    int32_t mode;
};

// 按输出坐标反推输入坐标，覆盖 DCR 与 CRD 两种官方模式。
__global__ void depth_to_space_kernel(
    const float* input,
    float* output,
    DepthToSpaceParams p,
    size_t out_len
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    int out_c = p.channels / (p.blocksize * p.blocksize);
    int out_h = p.height * p.blocksize;
    int out_w = p.width * p.blocksize;

    int w = (int)(tid % (size_t)out_w);
    int h = (int)((tid / (size_t)out_w) % (size_t)out_h);
    int c = (int)((tid / ((size_t)out_h * (size_t)out_w)) % (size_t)out_c);
    int n = (int)(tid / ((size_t)out_c * (size_t)out_h * (size_t)out_w));

    int in_h = h / p.blocksize;
    int in_w = w / p.blocksize;
    int dy = h % p.blocksize;
    int dx = w % p.blocksize;
    int block_offset = dy * p.blocksize + dx;
    int in_c = p.mode == 0
        ? block_offset * out_c + c
        : c * p.blocksize * p.blocksize + block_offset;

    size_t in_idx = ((size_t)n * (size_t)p.channels * (size_t)p.height * (size_t)p.width)
                  + ((size_t)in_c * (size_t)p.height * (size_t)p.width)
                  + ((size_t)in_h * (size_t)p.width)
                  + (size_t)in_w;
    output[tid] = input[in_idx];
}

// 顺序读取 `[N, C, H, W, blocksize, mode]` 参数。
static int read_depth_to_space_params(const char* params_path, DepthToSpaceParams* params) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t values[6];
    if (fread(values, sizeof(int32_t), 6, fp) != 6) {
        fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    fclose(fp);
    params->batch = values[0];
    params->channels = values[1];
    params->height = values[2];
    params->width = values[3];
    params->blocksize = values[4];
    params->mode = values[5];
    return params->batch > 0 && params->channels > 0 && params->height > 0
        && params->width > 0 && params->blocksize > 0
        && params->channels % (params->blocksize * params->blocksize) == 0;
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

    DepthToSpaceParams params;
    if (!read_depth_to_space_params(params_path, &params)) {
        return 1;
    }

    size_t in_len = (size_t)params.batch * (size_t)params.channels * (size_t)params.height * (size_t)params.width;
    size_t in_bytes = in_len * sizeof(float);
    size_t out_bytes = out_len * sizeof(float);

    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t ri = fread(h_input, sizeof(float), in_len, fi);
    fclose(fi);
    if (ri != in_len) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    depth_to_space_kernel<<<blocks, threads>>>(d_input, d_output, params, out_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t wo = fwrite(h_output, sizeof(float), out_len, fo);
    fclose(fo);
    if (wo != out_len) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
