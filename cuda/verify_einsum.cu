#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

struct EinsumParams {
    int32_t M;
    int32_t K;
    int32_t N;
};

// Egor Izmaylov: Function `einsum_ij_jk_to_ik_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void einsum_ij_jk_to_ik_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = M * N;
    if (tid >= total) return;

    int i = tid / N;
    int j = tid % N;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[(size_t)i * K + k] * B[(size_t)k * N + j];
    }
    C[tid] = sum;
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <A.bin> <B.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <A.bin> <B.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* a_path = argv[2];
    const char* b_path = argv[3];
    const char* p_path = argv[4];
    const char* out_path = argv[5];

    EinsumParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(EinsumParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if ((size_t)p.M * p.N != out_len) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }

    size_t a_len = (size_t)p.M * p.K;
    size_t b_len = (size_t)p.K * p.N;

    std::vector<float> h_A(a_len);
    std::vector<float> h_B(b_len);

    FILE* fa = fopen(a_path, "rb");
    FILE* fb = fopen(b_path, "rb");
    if (!fa || !fb) {
        fprintf(stderr, "open inputs failed\n");
        return 1;
    }

    if (fread(h_A.data(), sizeof(float), a_len, fa) != a_len) {
        fprintf(stderr, "read A failed\n");
        fclose(fa);
        fclose(fb);
        return 1;
    }
    if (fread(h_B.data(), sizeof(float), b_len, fb) != b_len) {
        fprintf(stderr, "read B failed\n");
        fclose(fa);
        fclose(fb);
        return 1;
    }
    fclose(fa);
    fclose(fb);

    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    cudaMalloc(&d_A, a_len * sizeof(float));
    cudaMalloc(&d_B, b_len * sizeof(float));
    cudaMalloc(&d_C, out_len * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), a_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), b_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    einsum_ij_jk_to_ik_kernel<<<blocks, threads>>>(d_A, d_B, d_C, p.M, p.K, p.N);
    cudaDeviceSynchronize();

    std::vector<float> h_C(out_len);
    cudaMemcpy(h_C.data(), d_C, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "open out failed\n");
        return 1;
    }
    if (fwrite(h_C.data(), sizeof(float), out_len, fo) != out_len) {
        fprintf(stderr, "write out failed\n");
        fclose(fo);
        return 1;
    }
    fclose(fo);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}