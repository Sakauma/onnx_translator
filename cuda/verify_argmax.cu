#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

struct ArgParams {
    int32_t M;
    int32_t N;
    int32_t axis;
    int32_t keepdims;
    int32_t select_last_index;
};

// Egor Izmaylov: Function `argmax_axis1_2d_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void argmax_axis1_2d_kernel(const float* x, int64_t* out, int M, int N) {
    int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= M) return;

    const float* row_ptr = x + (size_t)row * N;
    float best = row_ptr[0];
    int best_idx = 0;

    for (int j = 1; j < N; ++j) {
        float v = row_ptr[j];
        if (v > best) {
            best = v;
            best_idx = j;
        }
    }
    out[row] = (int64_t)best_idx;
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

    ArgParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(ArgParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (p.axis != 1 || p.keepdims != 0 || p.select_last_index != 0) {
        fprintf(stderr, "This verifier only supports axis=1, keepdims=0, select_last_index=0.\n");
        return 1;
    }
    if (out_len != (size_t)p.M) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    size_t in_len = (size_t)p.M * p.N;
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
    int64_t* d_out = NULL;

    cudaMalloc(&d_x, in_len * sizeof(float));
    cudaMalloc(&d_out, out_len * sizeof(int64_t));

    cudaMemcpy(d_x, h_x.data(), in_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (p.M + threads - 1) / threads;
    argmax_axis1_2d_kernel<<<blocks, threads>>>(d_x, d_out, p.M, p.N);
    cudaDeviceSynchronize();

    std::vector<int64_t> h_out(out_len);
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(int64_t), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open out failed\n");
        return 1;
    }
    if (fwrite(h_out.data(), sizeof(int64_t), out_len, fo) != out_len) {
        fprintf(stderr, "write out failed\n");
        fclose(fo);
        return 1;
    }
    fclose(fo);

    cudaFree(d_x);
    cudaFree(d_out);
    return 0;
}