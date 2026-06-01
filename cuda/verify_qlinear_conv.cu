#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ double read_param(const double* data, int size, int full_idx, int full_size,
                             int channel_idx, double default_value) {
    if (data == NULL || size <= 0) return default_value;
    if (size == 1) return data[0];
    if (size == full_size) return data[full_idx];
    return data[channel_idx % size];
}

__device__ long long read_int_param(const double* data, int size, int full_idx, int full_size,
                                    int channel_idx, long long default_value) {
    return llround(read_param(data, size, full_idx, full_size, channel_idx, (double)default_value));
}

__device__ uint8_t saturate_uint8(double value) {
    long long rounded = llrint(value);
    if (rounded < 0) return 0;
    if (rounded > 255) return 255;
    return (uint8_t)rounded;
}

__global__ void qlinear_conv_kernel(const double* X, const double* XScale, const double* XZeroPoint,
                                    const double* W, const double* WScale, const double* WZeroPoint,
                                    const double* YScale, const double* YZeroPoint, uint8_t* Y,
                                    int batch, int in_c, int in_h, int in_w,
                                    int out_c, int k_h, int k_w,
                                    int out_h, int out_w,
                                    int pad_t, int pad_l, int stride_h, int stride_w,
                                    int dil_h, int dil_w, int group,
                                    int x_scale_size, int x_zp_size,
                                    int w_scale_size, int w_zp_size,
                                    int y_scale_size, int y_zp_size) {
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
    int x_total = batch * in_c * in_h * in_w;
    int w_total = out_c * in_c_per_group * k_h * k_w;

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
                long long x_zp = read_int_param(XZeroPoint, x_zp_size, x_idx, x_total, ic, 0);
                long long w_zp = read_int_param(WZeroPoint, w_zp_size, w_idx, w_total, m, 0);
                acc += (x_val - x_zp) * (w_val - w_zp);
            }
        }
    }

    double x_scale = read_param(XScale, x_scale_size, 0, 1, 0, 1.0);
    double w_scale = read_param(WScale, w_scale_size,
                                m * in_c_per_group * k_h * k_w, w_total, m, 1.0);
    double y_scale = read_param(YScale, y_scale_size, idx, total_elements, m, 1.0);
    double y_zp = read_param(YZeroPoint, y_zp_size, idx, total_elements, m, 0.0);
    double q = ((double)acc * x_scale * w_scale) / y_scale + y_zp;
    Y[idx] = saturate_uint8(q);
}

int main(int argc, char** argv) {
    if (argc < 12) return 1;

    long long out_len = atoll(argv[1]);
    int p[22];
    FILE* fp = fopen(argv[10], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 22, fp);
    fclose(fp);

    int N = p[0], IC = p[1], IH = p[2], IW = p[3];
    int OC = p[4], KH = p[5], KW = p[6];
    int OH = p[7], OW = p[8];
    int group = p[15];
    int x_scale_size = p[16], x_zp_size = p[17];
    int w_scale_size = p[18], w_zp_size = p[19];
    int y_scale_size = p[20], y_zp_size = p[21];
    int in_c_per_group = IC / group;

    size_t size_x = (size_t)N * IC * IH * IW * sizeof(double);
    size_t size_w = (size_t)OC * in_c_per_group * KH * KW * sizeof(double);
    size_t size_x_scale = (size_t)x_scale_size * sizeof(double);
    size_t size_x_zp = (size_t)x_zp_size * sizeof(double);
    size_t size_w_scale = (size_t)w_scale_size * sizeof(double);
    size_t size_w_zp = (size_t)w_zp_size * sizeof(double);
    size_t size_y_scale = (size_t)y_scale_size * sizeof(double);
    size_t size_y_zp = (size_t)y_zp_size * sizeof(double);
    size_t size_y = (size_t)out_len * sizeof(uint8_t);

    double* h_x = (double*)malloc(size_x);
    double* h_x_scale = (double*)malloc(size_x_scale);
    double* h_x_zp = (double*)malloc(size_x_zp);
    double* h_w = (double*)malloc(size_w);
    double* h_w_scale = (double*)malloc(size_w_scale);
    double* h_w_zp = (double*)malloc(size_w_zp);
    double* h_y_scale = (double*)malloc(size_y_scale);
    double* h_y_zp = (double*)malloc(size_y_zp);
    uint8_t* h_y = (uint8_t*)malloc(size_y);

    FILE* f = fopen(argv[2], "rb"); fread(h_x, 1, size_x, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(h_x_scale, 1, size_x_scale, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(h_x_zp, 1, size_x_zp, f); fclose(f);
    f = fopen(argv[5], "rb"); fread(h_w, 1, size_w, f); fclose(f);
    f = fopen(argv[6], "rb"); fread(h_w_scale, 1, size_w_scale, f); fclose(f);
    f = fopen(argv[7], "rb"); fread(h_w_zp, 1, size_w_zp, f); fclose(f);
    f = fopen(argv[8], "rb"); fread(h_y_scale, 1, size_y_scale, f); fclose(f);
    f = fopen(argv[9], "rb"); fread(h_y_zp, 1, size_y_zp, f); fclose(f);

    double *d_x, *d_x_scale, *d_x_zp, *d_w, *d_w_scale, *d_w_zp, *d_y_scale, *d_y_zp;
    uint8_t* d_y;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x_scale, size_x_scale); cudaMemcpy(d_x_scale, h_x_scale, size_x_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x_zp, size_x_zp); cudaMemcpy(d_x_zp, h_x_zp, size_x_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w, size_w); cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w_scale, size_w_scale); cudaMemcpy(d_w_scale, h_w_scale, size_w_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w_zp, size_w_zp); cudaMemcpy(d_w_zp, h_w_zp, size_w_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y_scale, size_y_scale); cudaMemcpy(d_y_scale, h_y_scale, size_y_scale, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y_zp, size_y_zp); cudaMemcpy(d_y_zp, h_y_zp, size_y_zp, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);

    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    qlinear_conv_kernel<<<blocks, threads>>>(d_x, d_x_scale, d_x_zp, d_w, d_w_scale, d_w_zp,
        d_y_scale, d_y_zp, d_y, N, IC, IH, IW, OC, KH, KW, OH, OW,
        p[9], p[10], p[11], p[12], p[13], p[14], group,
        x_scale_size, x_zp_size, w_scale_size, w_zp_size, y_scale_size, y_zp_size);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    f = fopen(argv[11], "wb"); fwrite(h_y, 1, size_y, f); fclose(f);

    free(h_x); free(h_x_scale); free(h_x_zp); free(h_w); free(h_w_scale);
    free(h_w_zp); free(h_y_scale); free(h_y_zp); free(h_y);
    cudaFree(d_x); cudaFree(d_x_scale); cudaFree(d_x_zp); cudaFree(d_w); cudaFree(d_w_scale);
    cudaFree(d_w_zp); cudaFree(d_y_scale); cudaFree(d_y_zp); cudaFree(d_y);
    return 0;
}
