/**
  ******************************************************************************
  * @file        verify_dft.cu
  * @author      Egor Izmaylov
  * @brief       提供 DFT 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.04  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TWO_PI 6.283185307179586476925286766559
#define MAX_DFT_RANK 8

// 将多维坐标转换为行优先线性下标，用于按真实 rank/axis 独立验证 DFT 坐标映射。
__device__ size_t linear_index_from_coords(const int* coords, const int* shape, int rank) {
    size_t idx = 0;
    for (int d = 0; d < rank; ++d) {
        idx = idx * (size_t)shape[d] + (size_t)coords[d];
    }
    return idx;
}

// 读取末尾复数维中的实部或虚部；实数输入缺省虚部为 0。
__device__ double read_complex_component(
    const double* X,
    const int* input_shape,
    int rank,
    const int* coords,
    int component,
    int input_complex_dim
) {
    if (component == 1 && input_complex_dim == 1) return 0.0;
    int tmp[MAX_DFT_RANK];
    for (int d = 0; d < rank; ++d) tmp[d] = coords[d];
    tmp[rank - 1] = component;
    return X[linear_index_from_coords(tmp, input_shape, rank)];
}

// 实现 `dft_kernel` CUDA 参考 kernel，按真实 rank、axis 和末尾复数维执行朴素 DFT 公式。
__global__ void dft_kernel(
    const double* X,
    double* Y,
    int rank,
    const int* input_shape,
    const int* output_shape,
    int axis,
    int input_complex_dim,
    int output_complex_dim,
    int inverse,
    int onesided,
    int dft_length
) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = 1;
    for (int d = 0; d < rank; ++d) total *= (size_t)output_shape[d];
    if (idx >= total) return;

    int out_coords[MAX_DFT_RANK];
    int in_coords[MAX_DFT_RANK];
    size_t rem = idx;
    for (int d = rank - 1; d >= 0; --d) {
        out_coords[d] = (int)(rem % (size_t)output_shape[d]);
        rem /= (size_t)output_shape[d];
    }
    for (int d = 0; d < rank; ++d) in_coords[d] = out_coords[d];

    int component = out_coords[rank - 1];
    int k = out_coords[axis];
    int input_len = input_shape[axis];

    if (inverse && onesided) {
        double real_sum = 0.0;
        int max_freq = dft_length / 2;
        for (int f = 0; f < input_len; ++f) {
            if (f > max_freq) continue;
            in_coords[axis] = f;
            double xr = read_complex_component(X, input_shape, rank, in_coords, 0, input_complex_dim);
            double xi = read_complex_component(X, input_shape, rank, in_coords, 1, input_complex_dim);
            double angle = TWO_PI * (double)f * (double)k / (double)dft_length;
            double contribution = xr * cos(angle) - xi * sin(angle);
            if (f != 0 && !(dft_length % 2 == 0 && f == dft_length / 2)) {
                contribution *= 2.0;
            }
            real_sum += contribution;
        }
        if (component == 0) Y[idx] = real_sum / (double)dft_length;
        return;
    }

    double real_sum = 0.0;
    double imag_sum = 0.0;
    double sign = inverse ? 1.0 : -1.0;
    for (int n = 0; n < dft_length; ++n) {
        double xr = 0.0;
        double xi = 0.0;
        if (n < input_len) {
            in_coords[axis] = n;
            xr = read_complex_component(X, input_shape, rank, in_coords, 0, input_complex_dim);
            xi = read_complex_component(X, input_shape, rank, in_coords, 1, input_complex_dim);
        }
        double angle = sign * TWO_PI * (double)k * (double)n / (double)dft_length;
        double ca = cos(angle);
        double sa = sin(angle);
        real_sum += xr * ca - xi * sa;
        imag_sum += xr * sa + xi * ca;
    }

    if (inverse) {
        real_sum /= (double)dft_length;
        imag_sum /= (double)dft_length;
    }
    Y[idx] = component == 0 ? real_sum : imag_sum;
}

// 读取 int32 参数块，支持新版 rank/shape 协议，并兼容旧版 9 整数协议。
static int read_params(const char* path, int32_t** params, size_t* count) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return 0;
    }
    long bytes = ftell(fp);
    if (bytes <= 0 || bytes % (long)sizeof(int32_t) != 0) {
        fclose(fp);
        return 0;
    }
    rewind(fp);
    *count = (size_t)bytes / sizeof(int32_t);
    *params = (int32_t*)malloc((size_t)bytes);
    if (!*params) {
        fclose(fp);
        return 0;
    }
    size_t got = fread(*params, sizeof(int32_t), *count, fp);
    fclose(fp);
    return got == *count;
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <X.bin> <dft_length.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <X.bin> <dft_length.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t* p = NULL;
    size_t param_count = 0;
    if (!read_params(argv[4], &p, &param_count)) return 2;

    int rank = 0;
    int axis = 1;
    int inverse = 0;
    int onesided = 0;
    int dft_length = 0;
    int input_shape_host[MAX_DFT_RANK] = {0};
    int output_shape_host[MAX_DFT_RANK] = {0};

    if (param_count == 9) {
        rank = 3;
        input_shape_host[0] = p[0];
        input_shape_host[1] = p[1];
        input_shape_host[2] = p[2];
        output_shape_host[0] = p[0];
        output_shape_host[1] = p[3];
        output_shape_host[2] = p[4];
        axis = p[5];
        inverse = p[6];
        onesided = p[7];
        dft_length = p[8];
    } else {
        rank = p[0];
        size_t expected_count = (size_t)(7 + 2 * rank);
        if (rank <= 1 || rank > MAX_DFT_RANK || param_count != expected_count) {
            free(p);
            return 3;
        }
        axis = p[1];
        inverse = p[2];
        onesided = p[3];
        dft_length = p[4];
        for (int d = 0; d < rank; ++d) input_shape_host[d] = p[7 + d];
        for (int d = 0; d < rank; ++d) output_shape_host[d] = p[7 + rank + d];
    }
    free(p);

    if (axis < 0) axis += rank;
    int input_complex_dim = input_shape_host[rank - 1];
    int output_complex_dim = output_shape_host[rank - 1];
    if (axis < 0 || axis >= rank - 1 || dft_length <= 0) return 4;
    if ((input_complex_dim != 1 && input_complex_dim != 2) || (output_complex_dim != 1 && output_complex_dim != 2)) return 4;

    size_t x_len = 1;
    size_t expected_out_len = 1;
    for (int d = 0; d < rank; ++d) {
        x_len *= (size_t)input_shape_host[d];
        expected_out_len *= (size_t)output_shape_host[d];
    }
    if (out_len != expected_out_len) return 4;

    double* h_x = (double*)malloc(x_len * sizeof(double));
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_x || !h_y) return 5;

    FILE* fx = fopen(argv[2], "rb");
    if (!fx) return 6;
    fread(h_x, sizeof(double), x_len, fx);
    fclose(fx);

    double* d_x = NULL;
    double* d_y = NULL;
    int* d_input_shape = NULL;
    int* d_output_shape = NULL;
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMalloc(&d_input_shape, (size_t)rank * sizeof(int));
    cudaMalloc(&d_output_shape, (size_t)rank * sizeof(int));
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape_host, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shape, output_shape_host, (size_t)rank * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    dft_kernel<<<blocks, threads>>>(
        d_x,
        d_y,
        rank,
        d_input_shape,
        d_output_shape,
        axis,
        input_complex_dim,
        output_complex_dim,
        inverse,
        onesided,
        dft_length
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fo = fopen(argv[5], "wb");
    if (!fo) return 7;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_input_shape);
    cudaFree(d_output_shape);
    return 0;
}
