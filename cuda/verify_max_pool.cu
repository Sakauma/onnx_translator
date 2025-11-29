#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool_kernel(const double* X, double* Y,
                               int batch, int channels, int in_h, int in_w,
                               int out_h, int out_w,
                               int k_h, int k_w,
                               int pad_t, int pad_l, int str_h, int str_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int c  = temp % channels; temp /= channels;
    int n  = temp;

    double max_val = -DBL_MAX;

    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            int h_in = oh * str_h + kh - pad_t;
            int w_in = ow * str_w + kw - pad_l;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int x_idx = ((n * channels + c) * in_h + h_in) * in_w + w_in;
                double val = X[x_idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    Y[idx] = max_val;
}

int main(int argc, char** argv) {
    // argv: [1]out_size, [2]X.bin, [3]params.bin, [4]out.bin
    if (argc < 5) return 1;
    
    long long out_len = atoll(argv[1]);
    
    // Params: [N, C, IH, IW, OH, OW, KH, KW, pad_t, pad_l, str_h, str_w]
    int p[12];
    FILE *fp = fopen(argv[3], "rb"); fread(p, sizeof(int), 12, fp); fclose(fp);
    
    size_t size_x = p[0]*p[1]*p[2]*p[3] * sizeof(double);
    size_t size_y = out_len * sizeof(double);
    
    double *h_x = (double*)malloc(size_x);
    double *h_y = (double*)malloc(size_y);
    FILE *fx = fopen(argv[2], "rb"); fread(h_x, 1, size_x, fx); fclose(fx);
    
    double *d_x, *d_y;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    
    maxpool_kernel<<<(out_len+255)/256, 256>>>(d_x, d_y, 
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]);
        
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[4], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);
    
    free(h_x); free(h_y); cudaFree(d_x); cudaFree(d_y);
    return 0;
}