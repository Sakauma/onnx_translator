/**
  ******************************************************************************
  * @file        verify_bitcast.cu
  * @author      Egor Izmaylov
  * @brief       提供 BitCast 算子的 CUDA 参考验证程序，按原始字节验证等宽 dtype 重解释。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 按字节复制输入到输出，匹配 BitCast 保留底层位模式的官方语义。
__global__ void bitcast_copy_kernel(const uint8_t* input, uint8_t* output, size_t nbytes) {
    size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (t < nbytes) {
        output[t] = input[t];
    }
}

// 读取 params.bin 中记录的单元素字节数。
static int read_elem_size(const char* params_path) {
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return -1;
    }
    int elem_size = 0;
    size_t r = fread(&elem_size, sizeof(int), 1, fp);
    fclose(fp);
    if (r != 1 || elem_size <= 0) {
        fprintf(stderr, "read params failed\n");
        return -1;
    }
    return elem_size;
}

// 作为 CUDA 验证程序入口，读取原始输入字节、执行等宽复制并写回输出字节。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];
    int elem_size = read_elem_size(params_path);
    if (elem_size <= 0) return 1;

    size_t nbytes = n * (size_t)elem_size;
    uint8_t* h_input = (uint8_t*)malloc(nbytes);
    uint8_t* h_output = (uint8_t*)malloc(nbytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t r = fread(h_input, 1, nbytes, fi);
    fclose(fi);
    if (r != nbytes) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    uint8_t* d_input = NULL;
    uint8_t* d_output = NULL;
    cudaMalloc(&d_input, nbytes);
    cudaMalloc(&d_output, nbytes);
    cudaMemcpy(d_input, h_input, nbytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((nbytes + (size_t)threads - 1) / (size_t)threads);
    bitcast_copy_kernel<<<blocks, threads>>>(d_input, d_output, nbytes);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, nbytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_output, 1, nbytes, fo);
    fclose(fo);
    if (w != nbytes) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
