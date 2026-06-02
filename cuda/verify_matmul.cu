#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct MatmulParams {
    int M;
    int K;
    int N;
};

// Egor Izmaylov: Function `matmul_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row < M && col < N) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += (double)A[row * K + k] * (double)B[k * N + col];
        }
        C[row * N + col] = (float)acc;
    }
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <A.bin> <B.bin> <params.bin> <out.bin>
    if (argc != 6) {
        printf("Usage: %s <out_len> <A.bin> <B.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* a_path = argv[2];
    const char* b_path = argv[3];
    const char* p_path = argv[4];
    const char* out_path = argv[5];

    // 读参数
    MatmulParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) { printf("open params failed\n"); return 1; }
    size_t pr = fread(&p, sizeof(MatmulParams), 1, fp);
    fclose(fp);
    if (pr != 1) { printf("read params failed\n"); return 1; }

    int M = p.M, K = p.K, N = p.N;
    if ((size_t)(M * N) != out_len) {
        printf("out_len mismatch: got %zu expected %d\n", out_len, M * N);
        return 1;
    }

    size_t a_len = (size_t)M * (size_t)K;
    size_t b_len = (size_t)K * (size_t)N;
    size_t a_bytes = a_len * sizeof(float);
    size_t b_bytes = b_len * sizeof(float);
    size_t c_bytes = out_len * sizeof(float);

    // host
    float* hA = (float*)malloc(a_bytes);
    float* hB = (float*)malloc(b_bytes);
    float* hC = (float*)malloc(c_bytes);
    if (!hA || !hB || !hC) { printf("malloc failed\n"); return 1; }

    FILE* fa = fopen(a_path, "rb");
    FILE* fb = fopen(b_path, "rb");
    if (!fa || !fb) { printf("open input failed\n"); return 1; }
    size_t ra = fread(hA, sizeof(float), a_len, fa);
    size_t rb = fread(hB, sizeof(float), b_len, fb);
    fclose(fa); fclose(fb);
    if (ra != a_len || rb != b_len) { printf("fread mismatch\n"); return 1; }

    // device
    float *dA = NULL, *dB = NULL, *dC = NULL;
    cudaMalloc(&dA, a_bytes);
    cudaMalloc(&dB, b_bytes);
    cudaMalloc(&dC, c_bytes);

    cudaMemcpy(dA, hA, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, b_bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, c_bytes, cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open out failed\n"); return 1; }
    size_t wc = fwrite(hC, sizeof(float), out_len, fo);
    fclose(fo);
    if (wc != out_len) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
