#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

struct GatherNDParams {
    int32_t A; 
    int32_t B; 
    int32_t I; 
    int32_t K; 
};

// Egor Izmaylov: Function `gathernd_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void gathernd_kernel(
    const float* data,
    const int64_t* idx,
    float* out,
    int A,
    int B,
    int I
) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= I) return;

    int64_t r = idx[(size_t)i * 2 + 0];
    int64_t c = idx[(size_t)i * 2 + 1];

    if (r < 0) r += A;
    if (c < 0) c += B;

    if (r < 0) r = 0;
    if (r >= A) r = A - 1;
    if (c < 0) c = 0;
    if (c >= B) c = B - 1;

    out[i] = data[(size_t)r * B + (size_t)c];
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <data.bin> <indices.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* data_path = argv[2];
    const char* idx_path  = argv[3];
    const char* p_path    = argv[4];
    const char* out_path  = argv[5];

    GatherNDParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    size_t pr = fread(&p, sizeof(GatherNDParams), 1, fp);
    fclose(fp);
    if (pr != 1) {
        fprintf(stderr, "read params failed\n");
        return 1;
    }

    if (p.K != 2) {
        fprintf(stderr, "This verifier supports K==2 only.\n");
        return 1;
    }
    if (out_len != (size_t)p.I) {
        fprintf(stderr, "out_len mismatch: got %zu expected %d\n", out_len, p.I);
        return 1;
    }

    size_t data_len = (size_t)p.A * (size_t)p.B;
    size_t idx_len  = (size_t)p.I * (size_t)p.K;

    size_t data_bytes = data_len * sizeof(float);
    size_t idx_bytes  = idx_len * sizeof(int64_t);
    size_t out_bytes  = out_len * sizeof(float);

    float* h_data = (float*)malloc(data_bytes);
    int64_t* h_idx = (int64_t*)malloc(idx_bytes);
    float* h_out = (float*)malloc(out_bytes);
    if (!h_data || !h_idx || !h_out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fd = fopen(data_path, "rb");
    FILE* fi = fopen(idx_path, "rb");
    if (!fd || !fi) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    size_t rd = fread(h_data, sizeof(float), data_len, fd);
    size_t ri = fread(h_idx, sizeof(int64_t), idx_len, fi);
    fclose(fd);
    fclose(fi);
    if (rd != data_len || ri != idx_len) {
        fprintf(stderr, "fread mismatch\n");
        return 1;
    }

    float* d_data = NULL;
    int64_t* d_idx = NULL;
    float* d_out = NULL;

    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_idx, idx_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, idx_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((p.I + threads - 1) / threads);
    gathernd_kernel<<<blocks, threads>>>(d_data, d_idx, d_out, p.A, p.B, p.I);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

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

    cudaFree(d_data);
    cudaFree(d_idx);
    cudaFree(d_out);
    free(h_data);
    free(h_idx);
    free(h_out);
    return 0;
}