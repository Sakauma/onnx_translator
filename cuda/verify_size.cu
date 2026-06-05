/**
  ******************************************************************************
  * @file        verify_size.cu
  * @author      Egor Izmaylov
  * @brief       提供 Size 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 写出输入张量元素总数，匹配 ONNX Size 的标量 int64 输出语义。
__global__ void size_kernel(int64_t* output, int64_t input_size) {
    output[0] = input_size;
}

// 读取输入元素总数参数。
static int read_size_param(const char* params_path, int64_t* input_size) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    if (fread(input_size, sizeof(int64_t), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "read params failed\n");
        return 0;
    }
    fclose(fp);
    return *input_size >= 0;
}

// 作为 CUDA 验证程序入口，从参数文件读取输入规模、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* params_path = argv[3];
    const char* out_path = argv[4];
    if (out_len != 1) {
        fprintf(stderr, "Size verifier expects scalar output\n");
        return 1;
    }

    int64_t input_size = 0;
    if (!read_size_param(params_path, &input_size)) {
        return 1;
    }

    int64_t h_output = 0;
    int64_t* d_output = NULL;
    cudaMalloc(&d_output, sizeof(int64_t));
    size_kernel<<<1, 1>>>(d_output, input_size);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_output, d_output, sizeof(int64_t), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t wo = fwrite(&h_output, sizeof(int64_t), 1, fo);
    fclose(fo);
    if (wo != 1) {
        fprintf(stderr, "write output failed\n");
        return 1;
    }

    cudaFree(d_output);
    return 0;
}
