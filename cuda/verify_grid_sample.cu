/**
  ******************************************************************************
  * @file        verify_grid_sample.cu
  * @author      Egor Izmaylov
  * @brief       提供 GridSample 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>

struct GridSampleParams {
    int32_t batch;
    int32_t channels;
    int32_t height;
    int32_t width;
    int32_t out_h;
    int32_t out_w;
    int32_t mode;
    int32_t padding_mode;
    int32_t align_corners;
};

// 将 [-1, 1] 规范化坐标转换为输入图像实际坐标。
__device__ double grid_denormalize_ref(double coord, int length, int align_corners) {
    if (align_corners) {
        return (coord + 1.0) * (double)(length - 1) / 2.0;
    }
    return ((coord + 1.0) * (double)length - 1.0) / 2.0;
}

// 实现 ONNX reflection padding 使用的坐标反射。
__device__ double grid_reflect_coordinate_ref(double coord, double low, double high) {
    if (high <= low) return low;
    double span = high - low;
    double value = fabs(fmod(coord - low, 2.0 * span));
    if (value > span) value = 2.0 * span - value;
    return value + low;
}

// 根据 padding_mode 将坐标映射到可采样范围，zeros 模式保留原坐标。
__device__ double grid_sample_coordinate_ref(double coord, int length, int padding_mode, int align_corners) {
    if (padding_mode == 1) {
        return fmin(fmax(coord, 0.0), (double)(length - 1));
    }
    if (padding_mode == 2) {
        double low = align_corners ? 0.0 : -0.5;
        double high = align_corners ? (double)(length - 1) : (double)length - 0.5;
        double reflected = grid_reflect_coordinate_ref(coord, low, high);
        return fmin(fmax(reflected, 0.0), (double)(length - 1));
    }
    return coord;
}

// 按 NCHW 布局读取一个像素；zeros padding 下越界返回 0。
__device__ double grid_get_pixel_2d_ref(
    const double* input,
    GridSampleParams p,
    int n,
    int c,
    double y,
    double x
) {
    if (p.padding_mode == 1 || p.padding_mode == 2) {
        y = grid_sample_coordinate_ref(y, p.height, p.padding_mode, p.align_corners);
        x = grid_sample_coordinate_ref(x, p.width, p.padding_mode, p.align_corners);
    }
    int yi = (int)y;
    int xi = (int)x;
    if (yi < 0 || yi >= p.height || xi < 0 || xi >= p.width) return 0.0;
    size_t idx = ((size_t)n * p.channels * p.height * p.width)
               + ((size_t)c * p.height * p.width)
               + ((size_t)yi * p.width)
               + (size_t)xi;
    return input[idx];
}

// 计算双线性插值结果。
__device__ double grid_bilinear_sample_2d_ref(
    const double* input,
    GridSampleParams p,
    int n,
    int c,
    double y,
    double x
) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    double ly = y - (double)y0;
    double lx = x - (double)x0;
    double hy = 1.0 - ly;
    double hx = 1.0 - lx;
    return grid_get_pixel_2d_ref(input, p, n, c, y0, x0) * hy * hx
         + grid_get_pixel_2d_ref(input, p, n, c, y0, x1) * hy * lx
         + grid_get_pixel_2d_ref(input, p, n, c, y1, x0) * ly * hx
         + grid_get_pixel_2d_ref(input, p, n, c, y1, x1) * ly * lx;
}

// 计算 bicubic 插值权重。
__device__ void grid_cubic_coefficients_ref(double t, double coeffs[4]) {
    double alpha = -0.75;
    double x = fabs(t);
    coeffs[0] = ((alpha * (x + 1.0) - 5.0 * alpha) * (x + 1.0) + 8.0 * alpha) * (x + 1.0) - 4.0 * alpha;
    coeffs[1] = ((alpha + 2.0) * x - (alpha + 3.0)) * x * x + 1.0;
    coeffs[2] = ((alpha + 2.0) * (1.0 - x) - (alpha + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    coeffs[3] = ((alpha * (2.0 - x) - 5.0 * alpha) * (2.0 - x) + 8.0 * alpha) * (2.0 - x) - 4.0 * alpha;
}

// 计算 bicubic 插值结果。
__device__ double grid_bicubic_sample_2d_ref(
    const double* input,
    GridSampleParams p,
    int n,
    int c,
    double y,
    double x
) {
    int y0 = (int)floor(y);
    int x0 = (int)floor(x);
    double cy[4];
    double cx[4];
    grid_cubic_coefficients_ref(y - (double)y0, cy);
    grid_cubic_coefficients_ref(x - (double)x0, cx);
    double total = 0.0;
    for (int iy = 0; iy < 4; iy++) {
        for (int ix = 0; ix < 4; ix++) {
            total += cy[iy] * cx[ix] * grid_get_pixel_2d_ref(input, p, n, c, y0 - 1 + iy, x0 - 1 + ix);
        }
    }
    return total;
}

// GridSample CUDA reference kernel，输出布局为 [N, C, Hout, Wout]。
__global__ void grid_sample_kernel(const double* input, const double* grid, double* output, GridSampleParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int ox = (int)(tid % (size_t)p.out_w);
    int oy = (int)((tid / (size_t)p.out_w) % (size_t)p.out_h);
    int c = (int)((tid / ((size_t)p.out_w * p.out_h)) % (size_t)p.channels);
    int n = (int)(tid / ((size_t)p.out_w * p.out_h * p.channels));

    size_t grid_idx = ((size_t)n * p.out_h * p.out_w * 2) + ((size_t)oy * p.out_w * 2) + ((size_t)ox * 2);
    double x_norm = (double)grid[grid_idx];
    double y_norm = (double)grid[grid_idx + 1];
    double in_x = grid_denormalize_ref(x_norm, p.width, p.align_corners);
    double in_y = grid_denormalize_ref(y_norm, p.height, p.align_corners);

    double value = 0.0;
    if (p.mode == 1) {
        double sy = nearbyint(grid_sample_coordinate_ref(in_y, p.height, p.padding_mode, p.align_corners));
        double sx = nearbyint(grid_sample_coordinate_ref(in_x, p.width, p.padding_mode, p.align_corners));
        value = grid_get_pixel_2d_ref(input, p, n, c, sy, sx);
    } else if (p.mode == 2) {
        value = grid_bicubic_sample_2d_ref(input, p, n, c, in_y, in_x);
    } else {
        value = grid_bilinear_sample_2d_ref(input, p, n, c, in_y, in_x);
    }
    output[tid] = value;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <input.bin> <grid.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <grid.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* grid_path = argv[3];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    GridSampleParams p;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(GridSampleParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    size_t input_len = (size_t)p.batch * p.channels * p.height * p.width;
    size_t grid_len = (size_t)p.batch * p.out_h * p.out_w * 2;
    std::vector<double> h_input(input_len);
    std::vector<double> h_grid(grid_len);
    std::vector<double> h_out(out_len);

    fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed\n");
        return 1;
    }
    if (fread(h_input.data(), sizeof(double), input_len, fp) != input_len) {
        fprintf(stderr, "read input failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    fp = fopen(grid_path, "rb");
    if (!fp) {
        fprintf(stderr, "open grid failed\n");
        return 1;
    }
    if (fread(h_grid.data(), sizeof(double), grid_len, fp) != grid_len) {
        fprintf(stderr, "read grid failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    double* d_input = NULL;
    double* d_grid = NULL;
    double* d_out = NULL;
    cudaMalloc((void**)&d_input, input_len * sizeof(double));
    cudaMalloc((void**)&d_grid, grid_len * sizeof(double));
    cudaMalloc((void**)&d_out, out_len * sizeof(double));
    cudaMemcpy(d_input, h_input.data(), input_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, h_grid.data(), grid_len * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    grid_sample_kernel<<<blocks, threads>>>(d_input, d_grid, d_out, p, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(double), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open out failed\n");
        cudaFree(d_input);
        cudaFree(d_grid);
        cudaFree(d_out);
        return 1;
    }
    if (fwrite(h_out.data(), sizeof(double), out_len, fp) != out_len) {
        fprintf(stderr, "write out failed\n");
        fclose(fp);
        cudaFree(d_input);
        cudaFree(d_grid);
        cudaFree(d_out);
        return 1;
    }
    fclose(fp);

    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_out);
    return 0;
}
