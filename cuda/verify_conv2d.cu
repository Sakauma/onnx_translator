#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

// 辅助：读取 4D 数据 (带 padding 检查)
__device__ double get_val_4d(const double* data, int n, int c, int h, int w, 
                             int N, int C, int H, int W) {
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) {
        return 0.0;
    }
    return data[((n * C + c) * H + h) * W + w];
}

__global__ void conv2d_kernel(const double* X, const double* W, const double* B, double* Y,
                              int batch, int in_c, int in_h, int in_w,
                              int out_c, int k_h, int k_w,
                              int out_h, int out_w,
                              int pad_t, int pad_l, int stride_h, int stride_w, 
                              int dil_h, int dil_w, int group) {
    // 线程索引映射到输出 Y 的坐标 [n, m, oh, ow]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_c * out_h * out_w;
    
    if (idx >= total_elements) return;

    // 反解坐标
    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int m  = temp % out_c; temp /= out_c;
    int n  = temp;

    int in_c_per_group = in_c / group;
    int g = m / (out_c / group);

    double sum = 0.0;
    
    // 卷积累加
    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
        int ic = g * in_c_per_group + ic_g;
        for (int kh = 0; kh < k_h; kh++) {
            for (int kw = 0; kw < k_w; kw++) {
                int h_in = oh * stride_h + kh * dil_h - pad_t;
                int w_in = ow * stride_w + kw * dil_w - pad_l;
                
                double val_x = get_val_4d(X, n, ic, h_in, w_in, batch, in_c, in_h, in_w);
                
                // W: [out_c, in_c_per_group, k_h, k_w]
                int w_idx = ((m * in_c_per_group + ic_g) * k_h + kh) * k_w + kw;
                double val_w = W[w_idx];
                
                sum += val_x * val_w;
            }
        }
    }
    
    // Bias
    if (B != NULL) {
        sum += B[m];
    }
    
    Y[idx] = sum;
}

int main(int argc, char** argv) {
    if (argc < 6) return 1;

    long long out_len = atoll(argv[1]);
    
    // 读取 Params
    // layout: [N, IC, IH, IW, OC, KH, KW, OH, OW, pad_t, pad_l, str_h, str_w, dil_h, dil_w, group]
    int p[16];
    FILE *fp = fopen(argv[5], "rb");
    if(!fp) return 2;
    fread(p, sizeof(int), 16, fp);
    fclose(fp);
    
    int N=p[0], IC=p[1], IH=p[2], IW=p[3];
    int OC=p[4], KH=p[5], KW=p[6];
    int OH=p[7], OW=p[8];
    // p[9]..p[15] are conv params
    
    size_t size_x = N*IC*IH*IW * sizeof(double);
    size_t size_w = OC*(IC/p[15])*KH*KW * sizeof(double);
    size_t size_b = OC * sizeof(double);
    size_t size_y = N*OC*OH*OW * sizeof(double);
    
    double *h_x = (double*)malloc(size_x);
    double *h_w = (double*)malloc(size_w);
    double *h_b = NULL;
    double *h_y = (double*)malloc(size_y);
    
    FILE *fx = fopen(argv[2], "rb"); fread(h_x, 1, size_x, fx); fclose(fx);
    FILE *fw = fopen(argv[3], "rb"); fread(h_w, 1, size_w, fw); fclose(fw);
    
    if (strcmp(argv[4], "null") != 0) {
        h_b = (double*)malloc(size_b);
        FILE *fb = fopen(argv[4], "rb"); fread(h_b, 1, size_b, fb); fclose(fb);
    }
    
    double *d_x, *d_w, *d_b = NULL, *d_y;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w, size_w); cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    if (h_b) {
        cudaMalloc(&d_b, size_b); cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    }
    
    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    
    conv2d_kernel<<<blocks, threads>>>(d_x, d_w, d_b, d_y, 
        N, IC, IH, IW, OC, KH, KW, OH, OW, 
        p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
        
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[6], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);
    
    free(h_x); free(h_w); if(h_b) free(h_b); free(h_y);
    cudaFree(d_x); cudaFree(d_w); if(d_b) cudaFree(d_b); cudaFree(d_y);
    
    return 0;
}