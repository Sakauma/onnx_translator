/**
  ******************************************************************************
  * @file        verify_max_roi_pool.cu
  * @author      Egor Izmaylov
  * @brief       提供 MaxRoiPool 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.04  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 实现 `max_roi_pool_kernel` CUDA 参考 kernel，按 ONNX MaxRoiPool 的 ROI 分箱规则计算每个输出元素。
__global__ void max_roi_pool_kernel(
    const double* X,
    const double* rois,
    double* Y,
    int N,
    int C,
    int H,
    int W,
    int num_rois,
    int pooled_h,
    int pooled_w,
    float spatial_scale
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = num_rois * C * pooled_h * pooled_w;
    if (idx >= total) return;

    int t = idx;
    int pw = t % pooled_w; t /= pooled_w;
    int ph = t % pooled_h; t /= pooled_h;
    int c = t % C; t /= C;
    int roi_idx = t;

    const double* roi = rois + (size_t)roi_idx * 5;
    int batch = (int)roi[0];
    if (batch < 0 || batch >= N) {
        Y[idx] = 0.0;
        return;
    }

    int x1 = (int)nearbyint(roi[1] * (double)spatial_scale);
    int y1 = (int)nearbyint(roi[2] * (double)spatial_scale);
    int x2 = (int)nearbyint(roi[3] * (double)spatial_scale);
    int y2 = (int)nearbyint(roi[4] * (double)spatial_scale);
    int roi_w = max(x2 - x1 + 1, 1);
    int roi_h = max(y2 - y1 + 1, 1);
    double bin_h = (double)roi_h / (double)pooled_h;
    double bin_w = (double)roi_w / (double)pooled_w;

    int hstart = min(max((int)floor((double)ph * bin_h) + y1, 0), H);
    int hend = min(max((int)ceil((double)(ph + 1) * bin_h) + y1, 0), H);
    int wstart = min(max((int)floor((double)pw * bin_w) + x1, 0), W);
    int wend = min(max((int)ceil((double)(pw + 1) * bin_w) + x1, 0), W);

    if (hend <= hstart || wend <= wstart) {
        Y[idx] = 0.0;
        return;
    }

    double best = -DBL_MAX;
    for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
            double value = X[((size_t)batch * C + c) * H * W + (size_t)h * W + w];
            if (value > best) best = value;
        }
    }
    Y[idx] = best;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <X.bin> <rois.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <X.bin> <rois.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t p[7];
    float spatial_scale = 1.0f;
    FILE* fp = fopen(argv[4], "rb");
    if (!fp) return 2;
    if (fread(p, sizeof(int32_t), 7, fp) != 7) {
        fclose(fp);
        return 3;
    }
    if (fread(&spatial_scale, sizeof(float), 1, fp) != 1) {
        fclose(fp);
        return 4;
    }
    fclose(fp);

    int N = p[0], C = p[1], H = p[2], W = p[3];
    int num_rois = p[4], pooled_h = p[5], pooled_w = p[6];
    if (out_len != (size_t)num_rois * C * pooled_h * pooled_w) return 5;

    size_t x_len = (size_t)N * C * H * W;
    size_t rois_len = (size_t)num_rois * 5;
    double* h_x = (double*)malloc(x_len * sizeof(double));
    double* h_rois = (double*)malloc(rois_len * sizeof(double));
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_x || !h_rois || !h_y) return 6;

    FILE* fx = fopen(argv[2], "rb");
    FILE* fr = fopen(argv[3], "rb");
    if (!fx || !fr) return 7;
    fread(h_x, sizeof(double), x_len, fx);
    fread(h_rois, sizeof(double), rois_len, fr);
    fclose(fx);
    fclose(fr);

    double* d_x = NULL;
    double* d_rois = NULL;
    double* d_y = NULL;
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_rois, rois_len * sizeof(double));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rois, h_rois, rois_len * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    max_roi_pool_kernel<<<blocks, threads>>>(d_x, d_rois, d_y, N, C, H, W, num_rois, pooled_h, pooled_w, spatial_scale);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fo = fopen(argv[5], "wb");
    if (!fo) return 8;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_x);
    free(h_rois);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_rois);
    cudaFree(d_y);
    return 0;
}
