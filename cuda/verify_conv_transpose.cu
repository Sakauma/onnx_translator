/**
  ******************************************************************************
  * @file        verify_conv_transpose.cu
  * @author      Egor Izmaylov
  * @brief       提供 conv transpose 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// 实现 `conv_transpose_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void conv_transpose_kernel(const double* X, const double* W, const double* B, double* Y,
                                      int batch, int in_c, int in_h, int in_w,
                                      int m_per_group, int k_h, int k_w,
                                      int out_c, int out_h, int out_w,
                                      int pad_t, int pad_l, int stride_h, int stride_w,
                                      int dil_h, int dil_w, int group) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_c * out_h * out_w;
    if (idx >= total_elements) return;

    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int oc = temp % out_c; temp /= out_c;
    int n = temp;

    int in_c_per_group = in_c / group;
    int group_idx = oc / m_per_group;
    int oc_local = oc - group_idx * m_per_group;
    int ic_begin = group_idx * in_c_per_group;
    int ic_end = ic_begin + in_c_per_group;

    double sum = (B != NULL) ? B[oc] : 0.0;
    for (int ic = ic_begin; ic < ic_end; ic++) {
        for (int kh = 0; kh < k_h; kh++) {
            int h_offset = oh + pad_t - kh * dil_h;
            if (h_offset % stride_h != 0) continue;
            int ih = h_offset / stride_h;
            if (ih < 0 || ih >= in_h) continue;

            for (int kw = 0; kw < k_w; kw++) {
                int w_offset = ow + pad_l - kw * dil_w;
                if (w_offset % stride_w != 0) continue;
                int iw = w_offset / stride_w;
                if (iw < 0 || iw >= in_w) continue;

                int x_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                int w_idx = ((ic * m_per_group + oc_local) * k_h + kh) * k_w + kw;
                sum += X[x_idx] * W[w_idx];
            }
        }
    }

    Y[idx] = sum;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 6) return 1;

    long long out_len = atoll(argv[1]);
    int p[17];
    FILE* fp = fopen(argv[5], "rb");
    if (!fp) return 2;
    fread(p, sizeof(int), 17, fp);
    fclose(fp);

    int N = p[0], IC = p[1], IH = p[2], IW = p[3];
    int MPG = p[4], KH = p[5], KW = p[6];
    int OC = p[7], OH = p[8], OW = p[9];
    int group = p[16];

    size_t size_x = (size_t)N * IC * IH * IW * sizeof(double);
    size_t size_w = (size_t)IC * MPG * KH * KW * sizeof(double);
    size_t size_b = (size_t)OC * sizeof(double);
    size_t size_y = (size_t)N * OC * OH * OW * sizeof(double);

    double* h_x = (double*)malloc(size_x);
    double* h_w = (double*)malloc(size_w);
    double* h_b = NULL;
    double* h_y = (double*)malloc(size_y);

    FILE* fx = fopen(argv[2], "rb"); fread(h_x, 1, size_x, fx); fclose(fx);
    FILE* fw = fopen(argv[3], "rb"); fread(h_w, 1, size_w, fw); fclose(fw);
    if (strcmp(argv[4], "null") != 0) {
        h_b = (double*)malloc(size_b);
        FILE* fb = fopen(argv[4], "rb"); fread(h_b, 1, size_b, fb); fclose(fb);
    }

    double *d_x, *d_w, *d_b = NULL, *d_y;
    cudaMalloc(&d_x, size_x); cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w, size_w); cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, size_y);
    if (h_b) {
        cudaMalloc(&d_b, size_b);
        cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (out_len + threads - 1) / threads;
    conv_transpose_kernel<<<blocks, threads>>>(d_x, d_w, d_b, d_y,
        N, IC, IH, IW, MPG, KH, KW, OC, OH, OW,
        p[10], p[11], p[12], p[13], p[14], p[15], group);

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);
    FILE* fout = fopen(argv[6], "wb"); fwrite(h_y, 1, size_y, fout); fclose(fout);

    free(h_x); free(h_w); if (h_b) free(h_b); free(h_y);
    cudaFree(d_x); cudaFree(d_w); if (d_b) cudaFree(d_b); cudaFree(d_y);

    return 0;
}
