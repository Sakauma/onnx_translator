/**
  ******************************************************************************
  * @file        verify_gelu.cu
  * @author      Egor Izmaylov
  * @brief       提供 gelu 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  * @details     2026.06.05  V1.0.1  支持 ONNX approximate=tanh 近似模式
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Gelu 按 mode 选择精确 erf 公式或 ONNX tanh 近似公式。
__global__ void gelu_kernel(const float* input, float* output, size_t n, int approximate_mode) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < n) {
        float x = input[t];
        if (approximate_mode == 1) {
            output[t] = 0.5f * x * (
                1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))
            );
        } else {
            output[t] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
    }
}

// 读取近似模式参数；缺省时保持 ONNX 默认 approximate=none。
static int read_approximate_mode(const char* params_path, int* approximate_mode) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    size_t r = fread(approximate_mode, sizeof(int), 1, fp);
    fclose(fp);
    if (r != 1) {
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    return 1;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> [params.bin] <out.bin>
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> [params.bin] <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    int approximate_mode = 0;
    const char* out_path = argv[3];
    if (argc == 5) {
        if (!read_approximate_mode(argv[3], &approximate_mode)) {
            return 1;
        }
        out_path = argv[4];
    }

    size_t bytes = n * sizeof(float);
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t r = fread(h_input, sizeof(float), n, fi);
    fclose(fi);
    if (r != n) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    gelu_kernel<<<blocks, threads>>>(d_input, d_output, n, approximate_mode);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_output, sizeof(float), n, fo);
    fclose(fo);
    if (w != n) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
