/**
  ******************************************************************************
  * @file        verify_dequantize_linear.cu
  * @author      Egor Izmaylov
  * @brief       提供 dequantize linear 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 根据输出线性下标映射 per-tensor、per-axis 或已广播参数下标。
__device__ size_t qdq_param_index(size_t idx, size_t param_count, size_t output_count, int axis_dim, size_t axis_stride) {
    if (param_count <= 1) return 0;
    if (param_count == output_count) return idx;
    if (axis_dim > 0 && param_count == (size_t)axis_dim && axis_stride > 0) {
        return (idx / axis_stride) % (size_t)axis_dim;
    }
    return idx % param_count;
}

// 实现 `dequantize_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void dequantize_kernel(
    const double* x,
    const double* scale,
    const double* zp,
    double* out,
    size_t n,
    size_t scale_count,
    size_t zp_count,
    int axis_dim,
    size_t axis_stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t scale_idx = qdq_param_index(idx, scale_count, n, axis_dim, axis_stride);
        size_t zp_idx = qdq_param_index(idx, zp_count, n, axis_dim, axis_stride);
        out[idx] = (x[idx] - zp[zp_idx]) * scale[scale_idx];
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) return 1;
    size_t n = atol(argv[1]);
    size_t x_bytes = n * sizeof(double);
    const char* out_path = (argc == 7) ? argv[6] : argv[5];
    int rank = 1;
    int axis = 0;
    size_t scale_count = n;
    size_t zp_count = n;
    int axis_dim = 0;
    size_t axis_stride = 1;

    if (argc == 7) {
        FILE *fp = fopen(argv[5], "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long param_bytes = ftell(fp);
            rewind(fp);
            size_t param_count = (size_t)param_bytes / sizeof(int);
            int* params = (int*)malloc(param_count * sizeof(int));
            if (params && param_count > 0) {
                fread(params, sizeof(int), param_count, fp);
                if (param_count >= 4) {
                    rank = params[0];
                    axis = params[1];
                    scale_count = (size_t)params[2];
                    zp_count = (size_t)params[3];
                    if (rank > 0 && axis >= 0 && axis < rank && param_count >= (size_t)(4 + rank)) {
                        axis_dim = params[4 + axis];
                        axis_stride = 1;
                        for (int i = axis + 1; i < rank; ++i) {
                            axis_stride *= (size_t)params[4 + i];
                        }
                    }
                }
                free(params);
            }
            fclose(fp);
        }
    }
    if (scale_count == 0) scale_count = 1;
    if (zp_count == 0) zp_count = 1;
    
    size_t scale_bytes = scale_count * sizeof(double);
    size_t zp_bytes = zp_count * sizeof(double);
    double *h_x = (double*)malloc(x_bytes);
    double *h_s = (double*)calloc(scale_count, sizeof(double));
    double *h_z = (double*)calloc(zp_count, sizeof(double));
    double *h_out = (double*)malloc(x_bytes);
    
    FILE *fx = fopen(argv[2], "rb"); fread(h_x, 1, x_bytes, fx); fclose(fx);
    FILE *fs = fopen(argv[3], "rb"); fread(h_s, 1, scale_bytes, fs); fclose(fs);
    FILE *fz = fopen(argv[4], "rb"); fread(h_z, 1, zp_bytes, fz); fclose(fz);
    
    double *d_x, *d_s, *d_z, *d_out;
    cudaMalloc(&d_x, x_bytes); cudaMalloc(&d_s, scale_bytes); cudaMalloc(&d_z, zp_bytes); cudaMalloc(&d_out, x_bytes);
    
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, zp_bytes, cudaMemcpyHostToDevice);
    
    dequantize_kernel<<<(n + 255)/256, 256>>>(d_x, d_s, d_z, d_out, n, scale_count, zp_count, axis_dim, axis_stride);
    
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(out_path, "wb"); fwrite(h_out, 1, x_bytes, fout); fclose(fout);
    
    free(h_x); free(h_s); free(h_z); free(h_out);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_z); cudaFree(d_out);
    return 0;
}
