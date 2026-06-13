/**
  ******************************************************************************
  * @file        verify_det.cu
  * @author      Egor Izmaylov
  * @brief       提供 Det 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void det_kernel(const float* input, float* output, int32_t batch, int32_t n) {
    int b = (int)blockIdx.x;
    if (b >= batch || threadIdx.x != 0) return;

    const float* matrix = input + (size_t)b * (size_t)n * (size_t)n;
    double det = 0.0;
    if (n == 1) {
        det = matrix[0];
    } else if (n == 2) {
        det = (double)matrix[0] * matrix[3] - (double)matrix[1] * matrix[2];
    } else if (n == 3) {
        det =
            (double)matrix[0] * ((double)matrix[4] * matrix[8] - (double)matrix[5] * matrix[7])
            - (double)matrix[1] * ((double)matrix[3] * matrix[8] - (double)matrix[5] * matrix[6])
            + (double)matrix[2] * ((double)matrix[3] * matrix[7] - (double)matrix[4] * matrix[6]);
    }
    output[b] = (float)det;
}

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

    int32_t params[2] = {0, 0};
    FILE* fp = fopen(params_path, "rb");
    if (!fp || fread(params, sizeof(int32_t), 2, fp) != 2) {
        if (fp) fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 1;
    }
    fclose(fp);

    int32_t batch = params[0];
    int32_t n = params[1];
    if (batch <= 0 || n <= 0 || n > 3 || out_len != (size_t)batch) {
        fprintf(stderr, "invalid params\n");
        return 1;
    }

    size_t input_len = (size_t)batch * (size_t)n * (size_t)n;
    float* h_input = (float*)malloc(input_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi || fread(h_input, sizeof(float), input_len, fi) != input_len) {
        if (fi) fclose(fi);
        fprintf(stderr, "read input failed\n");
        return 1;
    }
    fclose(fi);

    float* d_input = NULL;
    float* d_output = NULL;
    cudaMalloc(&d_input, input_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMemcpy(d_input, h_input, input_len * sizeof(float), cudaMemcpyHostToDevice);

    det_kernel<<<batch, 1>>>(d_input, d_output, batch, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo || fwrite(h_output, sizeof(float), out_len, fo) != out_len) {
        if (fo) fclose(fo);
        fprintf(stderr, "write output failed\n");
        return 1;
    }
    fclose(fo);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
