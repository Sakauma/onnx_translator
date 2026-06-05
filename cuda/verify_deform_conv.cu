/**
  ******************************************************************************
  * @file        verify_deform_conv.cu
  * @author      Egor Izmaylov
  * @brief       提供 DeformConv 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

struct DeformConvParams {
    int32_t n;
    int32_t ic;
    int32_t ih;
    int32_t iw;
    int32_t oc;
    int32_t kh;
    int32_t kw;
    int32_t oh;
    int32_t ow;
    int32_t pad_h;
    int32_t pad_w;
    int32_t pad_h_end;
    int32_t pad_w_end;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t group;
    int32_t offset_group;
    int32_t has_bias;
    int32_t has_mask;
};

// 对实际图像坐标执行 zeros padding 的双线性采样。
__device__ double deform_conv_sample(const double* x, const DeformConvParams p, int n, int c, double y, double x_coord) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x_coord);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    double wy1 = y - (double)y0;
    double wx1 = x_coord - (double)x0;
    double wy0 = 1.0 - wy1;
    double wx0 = 1.0 - wx1;
    double value = 0.0;
    if (y0 >= 0 && y0 < p.ih && x0 >= 0 && x0 < p.iw) {
        size_t idx = ((size_t)n * (size_t)p.ic * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)c * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)y0 * (size_t)p.iw) + (size_t)x0;
        value += wy0 * wx0 * x[idx];
    }
    if (y0 >= 0 && y0 < p.ih && x1 >= 0 && x1 < p.iw) {
        size_t idx = ((size_t)n * (size_t)p.ic * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)c * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)y0 * (size_t)p.iw) + (size_t)x1;
        value += wy0 * wx1 * x[idx];
    }
    if (y1 >= 0 && y1 < p.ih && x0 >= 0 && x0 < p.iw) {
        size_t idx = ((size_t)n * (size_t)p.ic * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)c * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)y1 * (size_t)p.iw) + (size_t)x0;
        value += wy1 * wx0 * x[idx];
    }
    if (y1 >= 0 && y1 < p.ih && x1 >= 0 && x1 < p.iw) {
        size_t idx = ((size_t)n * (size_t)p.ic * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)c * (size_t)p.ih * (size_t)p.iw)
                   + ((size_t)y1 * (size_t)p.iw) + (size_t)x1;
        value += wy1 * wx1 * x[idx];
    }
    return value;
}

// 实现 DeformConv CUDA reference kernel，覆盖 2D group、offset group、bias 和 mask。
__global__ void deform_conv_kernel(
    const double* x,
    const double* w,
    const double* offset,
    const double* bias,
    const double* mask,
    double* y,
    DeformConvParams p,
    size_t total
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    size_t tmp = tid;
    int ow = (int)(tmp % (size_t)p.ow); tmp /= (size_t)p.ow;
    int oh = (int)(tmp % (size_t)p.oh); tmp /= (size_t)p.oh;
    int oc = (int)(tmp % (size_t)p.oc); tmp /= (size_t)p.oc;
    int n = (int)tmp;

    int in_per_group = p.ic / p.group;
    int out_per_group = p.oc / p.group;
    int in_per_offset_group = p.ic / p.offset_group;
    int conv_group = oc / out_per_group;
    int ic_begin = conv_group * in_per_group;
    int ic_end = ic_begin + in_per_group;

    double sum = p.has_bias ? bias[oc] : 0.0;
    for (int ic = ic_begin; ic < ic_end; ++ic) {
        int oc_local = ic - ic_begin;
        int offset_group_idx = ic / in_per_offset_group;
        for (int kh = 0; kh < p.kh; ++kh) {
            for (int kw = 0; kw < p.kw; ++kw) {
                int kernel_linear = kh * p.kw + kw;
                int offset_base_c = ((offset_group_idx * p.kh + kh) * p.kw + kw) * 2;
                size_t offset_h_idx = ((size_t)n * (size_t)(p.offset_group * p.kh * p.kw * 2) * (size_t)p.oh * (size_t)p.ow)
                                    + ((size_t)offset_base_c * (size_t)p.oh * (size_t)p.ow)
                                    + ((size_t)oh * (size_t)p.ow) + (size_t)ow;
                size_t offset_w_idx = offset_h_idx + (size_t)p.oh * (size_t)p.ow;
                double sample_y = -p.pad_h + oh * p.stride_h + kh * p.dilation_h + offset[offset_h_idx];
                double sample_x = -p.pad_w + ow * p.stride_w + kw * p.dilation_w + offset[offset_w_idx];
                double sampled = deform_conv_sample(x, p, n, ic, sample_y, sample_x);
                double mask_value = 1.0;
                if (p.has_mask) {
                    int mask_c = offset_group_idx * p.kh * p.kw + kernel_linear;
                    size_t mask_idx = ((size_t)n * (size_t)(p.offset_group * p.kh * p.kw) * (size_t)p.oh * (size_t)p.ow)
                                    + ((size_t)mask_c * (size_t)p.oh * (size_t)p.ow)
                                    + ((size_t)oh * (size_t)p.ow) + (size_t)ow;
                    mask_value = mask[mask_idx];
                }
                size_t w_idx = ((size_t)oc * (size_t)in_per_group * (size_t)p.kh * (size_t)p.kw)
                             + ((size_t)oc_local * (size_t)p.kh * (size_t)p.kw)
                             + ((size_t)kh * (size_t)p.kw) + (size_t)kw;
                sum += sampled * w[w_idx] * mask_value;
            }
        }
    }
    y[tid] = sum;
}

// 读取一个二进制文件到指定 vector 中。
template <typename T>
static int read_vector(const char* path, std::vector<T>& data) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t count = fread(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
    return count == data.size();
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <x.bin> <w.bin> <offset.bin> <bias.bin|null> <mask.bin|null> <params.bin> <out.bin>
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <w.bin> <offset.bin> <bias|null> <mask|null> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* w_path = argv[3];
    const char* offset_path = argv[4];
    const char* bias_path = argv[5];
    const char* mask_path = argv[6];
    const char* params_path = argv[7];
    const char* out_path = argv[8];

    DeformConvParams p;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) return 2;
    if (fread(&p, sizeof(DeformConvParams), 1, fp) != 1) {
        fclose(fp);
        return 3;
    }
    fclose(fp);

    if (out_len != (size_t)p.n * (size_t)p.oc * (size_t)p.oh * (size_t)p.ow) return 4;
    size_t x_len = (size_t)p.n * (size_t)p.ic * (size_t)p.ih * (size_t)p.iw;
    size_t w_len = (size_t)p.oc * (size_t)(p.ic / p.group) * (size_t)p.kh * (size_t)p.kw;
    size_t offset_len = (size_t)p.n * (size_t)(p.offset_group * p.kh * p.kw * 2) * (size_t)p.oh * (size_t)p.ow;
    size_t mask_len = (size_t)p.n * (size_t)(p.offset_group * p.kh * p.kw) * (size_t)p.oh * (size_t)p.ow;

    std::vector<double> h_x(x_len);
    std::vector<double> h_w(w_len);
    std::vector<double> h_offset(offset_len);
    std::vector<double> h_bias(p.has_bias ? (size_t)p.oc : 0);
    std::vector<double> h_mask(p.has_mask ? mask_len : 0);
    std::vector<double> h_out(out_len);
    if (!read_vector(x_path, h_x) || !read_vector(w_path, h_w) || !read_vector(offset_path, h_offset)) return 5;
    if (p.has_bias && (strcmp(bias_path, "null") == 0 || !read_vector(bias_path, h_bias))) return 6;
    if (p.has_mask && (strcmp(mask_path, "null") == 0 || !read_vector(mask_path, h_mask))) return 7;

    double *d_x = NULL, *d_w = NULL, *d_offset = NULL, *d_bias = NULL, *d_mask = NULL, *d_out = NULL;
    cudaMalloc((void**)&d_x, x_len * sizeof(double));
    cudaMalloc((void**)&d_w, w_len * sizeof(double));
    cudaMalloc((void**)&d_offset, offset_len * sizeof(double));
    cudaMalloc((void**)&d_out, out_len * sizeof(double));
    cudaMemcpy(d_x, h_x.data(), x_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), w_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset.data(), offset_len * sizeof(double), cudaMemcpyHostToDevice);
    if (p.has_bias) {
        cudaMalloc((void**)&d_bias, h_bias.size() * sizeof(double));
        cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (p.has_mask) {
        cudaMalloc((void**)&d_mask, h_mask.size() * sizeof(double));
        cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    deform_conv_kernel<<<blocks, threads>>>(d_x, d_w, d_offset, d_bias, d_mask, d_out, p, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(double), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) return 8;
    fwrite(h_out.data(), sizeof(double), out_len, fp);
    fclose(fp);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_offset);
    cudaFree(d_out);
    if (d_bias) cudaFree(d_bias);
    if (d_mask) cudaFree(d_mask);
    return 0;
}
