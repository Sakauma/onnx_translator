#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ long long read_zp(const double* zp, int zp_size, int full_idx, int channel_idx) {
    if (zp_size <= 0 || zp == NULL) return 0;
    if (zp_size == 1) return llround(zp[0]);
    return llround(zp[zp_size == 0 ? 0 : (channel_idx % zp_size)]);
}

__global__ void conv_integer_kernel(const double* X, const double* W,
                                    const double* XZeroPoint, const double* WZeroPoint,
                                    int32_t* Y,
                                    int batch, int in_c, int in_h, int in_w,
                                    int out_c, int k_h, int k_w,
                                    int out_h, int out_w,
                                    int pad_t, int pad_l, int stride_h, int stride_w,
                                    int dil_h, int dil_w, int group,
                                    int x_zp_size, int w_zp_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_c * out_h * out_w;
    if (idx >= total_elements) return;

    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int m = temp % out_c; temp /= out_c;
    int n = temp;

    int in_c_per_group = in_c / group;
    int out_c_per_group = out_c / group;
    int group_idx = m / out_c_per_group;

    long long acc = 0;
    for (int ic_g = 0; ic_g < in_c_per_group; ic_g++) {
        int ic = group_idx * in_c_per_group + ic_g;
        for (int kh = 0; kh < k_h; kh++) {
            for (int kw = 0; kw < k_w; kw++) {
                int ih = oh * stride_h + kh * dil_h - pad_t;
                int iw = ow * stride_w + kw * dil_w - pad_l;
                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;

                int x_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                int w_idx = ((m * in_c_per_group + ic_g) * k_h + kh) * k_w + kw;
                long long x_val = llround(X[x_idx]);
                long long w_val = llround(W[w_idx]);
                long long x_zp = (x_zp_size == batch * in_c * in_h * in_w) ? llround(XZeroPoint[x_idx]) : read_zp(XZeroPoint, x_zp_size, x_idx, ic);
                long long w_zp = (w_zp_size == out_c * in_c_per_group * k_h * k_w) ? llround(WZeroPoint[w_idx]) : read_zp(WZeroPoint, w_zp_size, w_idx, m);
                acc += (x_val - x_zp) * (w_val - w_zp);
            }
        }
    }

    Y[idx] = (int32_t)acc;
}

int main(int argc, char** argv) {
    if (argc < 8) return 1;

    long long out_len = atoll(argv[1]);
    int p[18];
    FILE* fp = fopen(argv[6], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 18, fp);
    fclose(fp);

    int N = p[0], IC = p[1], IH = p[2], IW = p[3];
    int OC = p[4], KH = p[5], KW = p[6];
    int OH = p[7], OW = p[8];
    int group = p[15], x_zp_size = p[16], w_zp_size = p[17];
    int in_c_per_group = IC / group;

    size_t size_x = (size_t)N * IC * IH * IW * sizeof(double);
    size_t size_w = (size_t)OC * in_c_per_group * KH * KW * sizeof(double);
    size_t size_x_zp = (size_t)x_zp_size * sizeof(double);
    size_t size_w_zp = (size_t)w_zp_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(int32_t);

    double* h_x = (double*)malloc(size_x);
    double* h_w = (double*)malloc(size_w);
    double* h_x_zp = (double*)malloc(size_x_zp);
    double* h_w_zp = (double*)malloc(size_w_zp);
    int32_t* h_y = (int32_t*)malloc(size_y);

    FILE* fx = fopen(argv[2], "rb"); fread(h_x, 1, size_x, fx); fclose(fx);
    FILE* fw = fopen(argv[3], "rb"); fread(h_w, 1, size_w, fw); fclose(fw);
    FILE* fxz = fopen(argv[4], "rb"); fread(h_x_zp, 1, size_x_zp, fxz); fclose(fxz);
    FILE* fwz = fopen(argv[5], "rb"); fread(h_w_zp, 1, size_w_zp, fwz); fclose(fwz);

    double *d_x, *d_w, *d_x_zp, *d_w_zp;
    int32_t* d_y;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w, size_w); cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x_zp, size_x_zp); cudaMemcpy(d_x_zp, h_x_zp, size_x_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w_zp, size_w_zp); cudaMemcpy(d_w_zp, h_w_zp, size_w_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);

    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    conv_integer_kernel<<<blocks, threads>>>(d_x, d_w, d_x_zp, d_w_zp, d_y,
        N, IC, IH, IW, OC, KH, KW, OH, OW,
        p[9], p[10], p[11], p[12], p[13], p[14], group, x_zp_size, w_zp_size);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE* fout = fopen(argv[7], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);

    free(h_x); free(h_w); free(h_x_zp); free(h_w_zp); free(h_y);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_x_zp); cudaFree(d_w_zp); cudaFree(d_y);
    return 0;
}
