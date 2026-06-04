/**
  ******************************************************************************
  * @file        verify_roi_align.cu
  * @author      Egor Izmaylov
  * @brief       提供 RoiAlign 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 按 ONNX RoiAlign 的边界规则读取一个双线性采样点，avg 模式返回四邻点加权和，max 模式返回四个加权项的最大值。
__device__ double roi_align_sample(const double* X, int N, int C, int H, int W, int batch, int c, double y, double x, int mode) {
    (void)N;
    if (y < -1.0 || y > (double)H || x < -1.0 || x > (double)W) {
        return 0.0;
    }

    y = y < 0.0 ? 0.0 : y;
    x = x < 0.0 ? 0.0 : x;
    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= H - 1) {
        y_high = y_low = H - 1;
        y = (double)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= W - 1) {
        x_high = x_low = W - 1;
        x = (double)x_low;
    } else {
        x_high = x_low + 1;
    }

    double ly = y - (double)y_low;
    double lx = x - (double)x_low;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;

    size_t base = ((size_t)batch * C + c) * H * W;
    double v1 = X[base + (size_t)y_low * W + x_low] * hy * hx;
    double v2 = X[base + (size_t)y_low * W + x_high] * hy * lx;
    double v3 = X[base + (size_t)y_high * W + x_low] * ly * hx;
    double v4 = X[base + (size_t)y_high * W + x_high] * ly * lx;

    if (mode == 1) {
        double m1 = v1 > v2 ? v1 : v2;
        double m2 = v3 > v4 ? v3 : v4;
        return m1 > m2 ? m1 : m2;
    }
    return v1 + v2 + v3 + v4;
}

// 实现 `roi_align_kernel` CUDA 参考 kernel，将每个输出元素映射到 ROI、通道和池化分箱。
__global__ void roi_align_kernel(
    const double* X,
    const double* rois,
    const int64_t* batch_indices,
    double* Y,
    int N,
    int C,
    int H,
    int W,
    int num_rois,
    int output_h,
    int output_w,
    int sampling_ratio,
    int mode,
    int coord_mode,
    float spatial_scale
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = num_rois * C * output_h * output_w;
    if (idx >= total) return;

    int t = idx;
    int pw = t % output_w; t /= output_w;
    int ph = t % output_h; t /= output_h;
    int c = t % C; t /= C;
    int roi_idx = t;

    int batch = (int)batch_indices[roi_idx];
    if (batch < 0 || batch >= N) {
        Y[idx] = 0.0;
        return;
    }

    const double* roi = rois + (size_t)roi_idx * 4;
    double offset = coord_mode == 0 ? 0.5 : 0.0;
    double roi_start_w = roi[0] * (double)spatial_scale - offset;
    double roi_start_h = roi[1] * (double)spatial_scale - offset;
    double roi_end_w = roi[2] * (double)spatial_scale - offset;
    double roi_end_h = roi[3] * (double)spatial_scale - offset;
    double roi_w = roi_end_w - roi_start_w;
    double roi_h = roi_end_h - roi_start_h;

    if (coord_mode != 0) {
        roi_w = roi_w > 1.0 ? roi_w : 1.0;
        roi_h = roi_h > 1.0 ? roi_h : 1.0;
    }

    double bin_h = roi_h / (double)output_h;
    double bin_w = roi_w / (double)output_w;
    int grid_h = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_h / (double)output_h);
    int grid_w = sampling_ratio > 0 ? sampling_ratio : (int)ceil(roi_w / (double)output_w);
    grid_h = grid_h > 1 ? grid_h : 1;
    grid_w = grid_w > 1 ? grid_w : 1;
    int count = grid_h * grid_w;

    double out = mode == 1 ? -DBL_MAX : 0.0;
    for (int iy = 0; iy < grid_h; ++iy) {
        double yy = roi_start_h + (double)ph * bin_h + ((double)iy + 0.5) * bin_h / (double)grid_h;
        for (int ix = 0; ix < grid_w; ++ix) {
            double xx = roi_start_w + (double)pw * bin_w + ((double)ix + 0.5) * bin_w / (double)grid_w;
            double value = roi_align_sample(X, N, C, H, W, batch, c, yy, xx, mode);
            if (mode == 1) {
                if (value > out) out = value;
            } else {
                out += value;
            }
        }
    }

    Y[idx] = mode == 1 ? out : out / (double)count;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <X.bin> <rois.bin> <batch_indices.bin> <params.bin> <out.bin>
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <out_len> <X.bin> <rois.bin> <batch_indices.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t p[10];
    float spatial_scale = 1.0f;
    FILE* fp = fopen(argv[5], "rb");
    if (!fp) return 2;
    if (fread(p, sizeof(int32_t), 10, fp) != 10) {
        fclose(fp);
        return 3;
    }
    if (fread(&spatial_scale, sizeof(float), 1, fp) != 1) {
        fclose(fp);
        return 4;
    }
    fclose(fp);

    int N = p[0], C = p[1], H = p[2], W = p[3];
    int num_rois = p[4], output_h = p[5], output_w = p[6];
    int sampling_ratio = p[7], mode = p[8], coord_mode = p[9];
    if (out_len != (size_t)num_rois * C * output_h * output_w) return 5;

    size_t x_len = (size_t)N * C * H * W;
    size_t rois_len = (size_t)num_rois * 4;
    double* h_x = (double*)malloc(x_len * sizeof(double));
    double* h_rois = (double*)malloc(rois_len * sizeof(double));
    int64_t* h_batch = (int64_t*)malloc((size_t)num_rois * sizeof(int64_t));
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_x || !h_rois || !h_batch || !h_y) return 6;

    FILE* fx = fopen(argv[2], "rb");
    FILE* fr = fopen(argv[3], "rb");
    FILE* fb = fopen(argv[4], "rb");
    if (!fx || !fr || !fb) return 7;
    fread(h_x, sizeof(double), x_len, fx);
    fread(h_rois, sizeof(double), rois_len, fr);
    fread(h_batch, sizeof(int64_t), (size_t)num_rois, fb);
    fclose(fx);
    fclose(fr);
    fclose(fb);

    double* d_x = NULL;
    double* d_rois = NULL;
    int64_t* d_batch = NULL;
    double* d_y = NULL;
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_rois, rois_len * sizeof(double));
    cudaMalloc(&d_batch, (size_t)num_rois * sizeof(int64_t));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rois, h_rois, rois_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch, h_batch, (size_t)num_rois * sizeof(int64_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    roi_align_kernel<<<blocks, threads>>>(
        d_x,
        d_rois,
        d_batch,
        d_y,
        N,
        C,
        H,
        W,
        num_rois,
        output_h,
        output_w,
        sampling_ratio,
        mode,
        coord_mode,
        spatial_scale
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fo = fopen(argv[6], "wb");
    if (!fo) return 8;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_x);
    free(h_rois);
    free(h_batch);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_rois);
    cudaFree(d_batch);
    cudaFree(d_y);
    return 0;
}
