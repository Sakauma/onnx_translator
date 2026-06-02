/*
 * 文件功能：提供 cumsum 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
 * 作者：Egor Izmaylov
 * 时间：2026-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

struct CumSumParams {
    int32_t N;          // length
    int32_t exclusive;  // 0/1
    int32_t reverse;    // 0/1
};

// 实现 `cumsum_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void cumsum_kernel(const float* in, float* out, int N, int exclusive, int reverse) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double acc = 0.0;

    if (!reverse) {
        for (int i = 0; i < N; ++i) {
            double x = (double)in[i];
            if (exclusive) {
                out[i] = (float)acc;
                acc += x;
            } else {
                acc += x;
                out[i] = (float)acc;
            }
        }
    } else {
        for (int ii = 0; ii < N; ++ii) {
            int i = N - 1 - ii;
            double x = (double)in[i];
            if (exclusive) {
                out[i] = (float)acc;
                acc += x;
            } else {
                acc += x;
                out[i] = (float)acc;
            }
        }
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <in.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <in.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* in_path = argv[2];
    const char* params_path = argv[3];
    const char* out_path = argv[4];

    CumSumParams p;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    size_t pr = fread(&p, sizeof(CumSumParams), 1, fp);
    fclose(fp);
    if (pr != 1) {
        fprintf(stderr, "read params failed\n");
        return 1;
    }

    if (p.N <= 0 || out_len != (size_t)p.N) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    size_t bytes = out_len * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(in_path, "rb");
    if (!fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t r = fread(h_in, sizeof(float), out_len, fi);
    fclose(fi);
    if (r != out_len) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float* d_in = NULL;
    float* d_out = NULL;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    cumsum_kernel<<<1, 1>>>(d_in, d_out, p.N, p.exclusive, p.reverse);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    size_t w = fwrite(h_out, sizeof(float), out_len, fo);
    fclose(fo);
    if (w != out_len) {
        fprintf(stderr, "fwrite mismatch\n");
        return 1;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
