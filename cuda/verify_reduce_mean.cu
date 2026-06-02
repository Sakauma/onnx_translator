#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct ReduceMeanParams {
    int M;
    int N;
};

// Egor Izmaylov: Function `sum_all_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void sum_all_kernel(const float* x, double* partial, int total) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double v = 0.0;
    if (idx < total) v = (double)x[idx];
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Egor Izmaylov: Function `sum_partial_kernel` is a CUDA reference kernel for the verifier; it maps thread indices to tensor elements and computes the expected GPU result.
__global__ void sum_partial_kernel(const double* partial, double* out, int nblocks) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double v = 0.0;

    // nblocks 一般不大，先简单累加
    for (int i = tid; i < nblocks; i += blockDim.x) v += partial[i];
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[0] = sdata[0];
}

// Egor Izmaylov: Function `main` is the standalone CUDA verifier entry point; it reads binary tensors, runs the reference calculation, and writes outputs for numerical_correctness.py.
int main(int argc, char** argv) {
    // <out_len> <in0.bin> <params.bin> <out.bin>
    if (argc != 5) {
        printf("Usage: %s <out_len> <in0.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* in_path = argv[2];
    const char* p_path  = argv[3];
    const char* out_path = argv[4];

    if (out_len != 1) {
        printf("Expected out_len=1 for reduce_mean(all), got %zu\n", out_len);
        return 1;
    }

    // 读参数 M,N
    ReduceMeanParams p;
    FILE* fp = fopen(p_path, "rb");
    if (!fp) { printf("open params failed\n"); return 1; }
    size_t pr = fread(&p, sizeof(ReduceMeanParams), 1, fp);
    fclose(fp);
    if (pr != 1) { printf("read params failed\n"); return 1; }

    int M = p.M, N = p.N;
    int total = M * N;
    if (total <= 0) { printf("invalid total\n"); return 1; }

    size_t in_bytes = (size_t)total * sizeof(float);

    // host
    float* h_x = (float*)malloc(in_bytes);
    float  h_y = 0.0f;
    if (!h_x) { printf("malloc failed\n"); return 1; }

    FILE* fi = fopen(in_path, "rb");
    if (!fi) { printf("open input failed\n"); return 1; }
    size_t rx = fread(h_x, sizeof(float), (size_t)total, fi);
    fclose(fi);
    if (rx != (size_t)total) { printf("fread mismatch\n"); return 1; }

    // device
    float* d_x = NULL;
    cudaMalloc(&d_x, in_bytes);
    cudaMemcpy(d_x, h_x, in_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    double* d_partial = NULL;
    double* d_sum = NULL;
    cudaMalloc(&d_partial, (size_t)blocks * sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));

    sum_all_kernel<<<blocks, threads>>>(d_x, d_partial, total);
    sum_partial_kernel<<<1, threads>>>(d_partial, d_sum, blocks);
    cudaDeviceSynchronize();

    double h_sum = 0.0;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    h_y = (float)(h_sum / (double)total);

    // 写输出
    FILE* fo = fopen(out_path, "wb");
    if (!fo) { printf("open output failed\n"); return 1; }
    size_t wy = fwrite(&h_y, sizeof(float), 1, fo);
    fclose(fo);
    if (wy != 1) { printf("fwrite mismatch\n"); return 1; }

    cudaFree(d_x);
    cudaFree(d_partial);
    cudaFree(d_sum);
    free(h_x);
    return 0;
}
