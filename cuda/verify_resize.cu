#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

struct ResizeParams {
    int32_t N;
    int32_t C;
    int32_t IH;
    int32_t IW;
    int32_t OH;
    int32_t OW;
};

// Egor Izmaylov: Function `resize_nearest_nchw_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void resize_nearest_nchw_kernel(
    const float* x,
    float* y,
    int N, int C, int IH, int IW, int OH, int OW
) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = N * C * OH * OW;
    if (tid >= total) return;

    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int c  = (tid / (OW * OH)) % C;
    int n  = tid / (OW * OH * C);

    float in_y = ((float)oh) * (float)IH / (float)OH;
    float in_x = ((float)ow) * (float)IW / (float)OW;

    int iy = (int)floorf(in_y);
    int ix = (int)floorf(in_x);

    if (iy < 0) iy = 0;
    if (iy >= IH) iy = IH - 1;
    if (ix < 0) ix = 0;
    if (ix >= IW) ix = IW - 1;

    size_t in_off = ((size_t)n * C + c) * IH * IW + (size_t)iy * IW + ix;
    size_t out_off = ((size_t)n * C + c) * OH * OW + (size_t)oh * OW + ow;
    y[out_off] = x[in_off];
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <x.bin> <params.bin> <out.bin>
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* p_path = argv[3];
    const char* out_path = argv[4];

    ResizeParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(ResizeParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    size_t in_len = (size_t)p.N * p.C * p.IH * p.IW;
    size_t expect_out = (size_t)p.N * p.C * p.OH * p.OW;
    if (expect_out != out_len) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    std::vector<float> h_x(in_len);

    FILE* fx = fopen(x_path, "rb");
    if (!fx) {
        fprintf(stderr, "open x failed\n");
        return 1;
    }
    if (fread(h_x.data(), sizeof(float), in_len, fx) != in_len) {
        fprintf(stderr, "read x failed\n");
        fclose(fx);
        return 1;
    }
    fclose(fx);

    float* d_x = NULL;
    float* d_y = NULL;

    cudaMalloc(&d_x, in_len * sizeof(float));
    cudaMalloc(&d_y, out_len * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), in_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    resize_nearest_nchw_kernel<<<blocks, threads>>>(d_x, d_y, p.N, p.C, p.IH, p.IW, p.OH, p.OW);
    cudaDeviceSynchronize();

    std::vector<float> h_y(out_len);
    cudaMemcpy(h_y.data(), d_y, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open out failed\n");
        return 1;
    }
    if (fwrite(h_y.data(), sizeof(float), out_len, fo) != out_len) {
        fprintf(stderr, "write out failed\n");
        fclose(fo);
        return 1;
    }
    fclose(fo);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}