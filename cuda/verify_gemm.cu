#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const double* A, const double* B, const double* C, double* Y,
                            int M, int N, int K,
                            float alpha, float beta, int transA, int transB,
                            int c_type) { // c_type: 0=None, 1=Scalar, 2=Row, 3=Col, 4=Matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    
    int m = idx / N;
    int n = idx % N;
    
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        int idx_a = transA ? (k * M + m) : (m * K + k);
        int idx_b = transB ? (n * K + k) : (k * N + n);
        sum += A[idx_a] * B[idx_b];
    }
    
    double res = alpha * sum;
    
    if (C != NULL) {
        double val_c = 0.0;
        if (c_type == 1) val_c = C[0];
        else if (c_type == 2) val_c = C[n]; // (1, N) or (N)
        else if (c_type == 3) val_c = C[m]; // (M, 1)
        else if (c_type == 4) val_c = C[m * N + n]; // (M, N)
        res += beta * val_c;
    }
    
    Y[idx] = res;
}

int main(int argc, char** argv) {
    // argv: [1]size, [2]A, [3]B, [4]C(or null), [5]params, [6]out
    if (argc < 6) return 1;
    
    long long out_len = atoll(argv[1]);
    
    // Params: [M, N, K, transA, transB, c_type, has_c] + alpha(float), beta(float)
    // alpha/beta 读取比较麻烦，我们约定 params.bin 前 7 个是 int，后面是 2 个 float
    struct {
        int M, N, K, transA, transB, c_type, has_c;
        float alpha, beta;
    } p;
    
    FILE *fp = fopen(argv[5], "rb"); fread(&p, sizeof(p), 1, fp); fclose(fp);
    
    size_t size_a = p.M * p.K * sizeof(double); // approx, depends on trans
    size_t size_b = p.K * p.N * sizeof(double);
    size_t size_y = out_len * sizeof(double);
    
    double *h_a = (double*)malloc(size_a);
    double *h_b = (double*)malloc(size_b);
    double *h_c = NULL;
    double *h_y = (double*)malloc(size_y);
    
    FILE *fa = fopen(argv[2], "rb"); fread(h_a, 1, size_a, fa); fclose(fa);
    FILE *fb = fopen(argv[3], "rb"); fread(h_b, 1, size_b, fb); fclose(fb);
    
    if (p.has_c) {
        // C size depends on type, simplifiction: calculate roughly or read all
        // We can seek file size
        FILE *fc = fopen(argv[4], "rb");
        fseek(fc, 0, SEEK_END); long sz = ftell(fc); fseek(fc, 0, SEEK_SET);
        h_c = (double*)malloc(sz);
        fread(h_c, 1, sz, fc); fclose(fc);
        // GPU malloc logic would need precise size, but cudaMalloc size just needs to be >=
    }
    
    double *d_a, *d_b, *d_c = NULL, *d_y;
    cudaMalloc(&d_a, size_a); cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size_b); cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    if (h_c) {
        // Simple hack: allocate M*N size max
        cudaMalloc(&d_c, p.M * p.N * sizeof(double)); 
        // We need to know exact size to copy? Not necessarily if we allocate enough.
        // Let's rely on c_type
        size_t sz_c = 0;
        if(p.c_type==1) sz_c=1;
        else if(p.c_type==2) sz_c=p.N;
        else if(p.c_type==3) sz_c=p.M;
        else sz_c=p.M*p.N;
        cudaMemcpy(d_c, h_c, sz_c*sizeof(double), cudaMemcpyHostToDevice);
    }
    
    gemm_kernel<<<(out_len+255)/256, 256>>>(d_a, d_b, d_c, d_y, 
        p.M, p.N, p.K, p.alpha, p.beta, p.transA, p.transB, p.c_type);
        
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[6], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);
    
    free(h_a); 
    free(h_b); 
    if (h_c) free(h_c); 
    free(h_y);
    
    cudaFree(d_a); 
    cudaFree(d_b); 
    if (d_c) cudaFree(d_c); 
    cudaFree(d_y);
    
    return 0;
}