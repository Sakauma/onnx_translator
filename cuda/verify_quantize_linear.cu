/**
  ******************************************************************************
  * @file        verify_quantize_linear.cu
  * @author      Egor Izmaylov
  * @brief       提供 quantize linear 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 升级: 所有指针和计算改为 double
// 实现 `saturate_cast_int8` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ double saturate_cast_int8(double val) {
    if (val > 127.0) return 127.0;
    if (val < -128.0) return -128.0;
    return val;
}

// 实现 `saturate_cast_uint8` 的 CUDA 验证辅助逻辑，为参考计算准备参数或中间结果。
__device__ double saturate_cast_uint8(double val) {
    if (val > 255.0) return 255.0;
    if (val < 0.0) return 0.0;
    return val;
}

// 实现 `quantize_kernel` CUDA 参考 kernel，将线程索引映射到张量元素并计算期望输出。
__global__ void quantize_kernel(const double* x, const double* scale, const double* zp, double* out, size_t n, int is_signed, int use_float_math) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double s = scale[idx];
        double z = zp[idx];
        double res = z;
        if (use_float_math) {
            float sf = (float)s;
            if (sf != 0.0f) {
                res = (double)rintf((float)x[idx] / sf) + z;
            }
        } else if (s != 0.0) {
            res = rint(x[idx] / s) + z;
        }
        
        if (is_signed) {
            out[idx] = saturate_cast_int8(res);
        } else {
            out[idx] = saturate_cast_uint8(res);
        }
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    if (argc < 7) return 1; 
    size_t n = atol(argv[1]);
    size_t bytes = n * sizeof(double);
    
    int is_signed = 1;
    int use_float_math = 1;
    FILE *fp = fopen(argv[5], "rb");
    if (fp) {
        fread(&is_signed, sizeof(int), 1, fp);
        fread(&use_float_math, sizeof(int), 1, fp);
        fclose(fp);
    }

    double *h_x = (double*)malloc(bytes);
    double *h_s = (double*)malloc(bytes);
    double *h_z = (double*)malloc(bytes);
    double *h_out = (double*)malloc(bytes);
    
    FILE *fx = fopen(argv[2], "rb"); fread(h_x, 1, bytes, fx); fclose(fx);
    FILE *fs = fopen(argv[3], "rb"); fread(h_s, 1, bytes, fs); fclose(fs);
    FILE *fz = fopen(argv[4], "rb"); fread(h_z, 1, bytes, fz); fclose(fz);
    
    double *d_x, *d_s, *d_z, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_s, bytes); cudaMalloc(&d_z, bytes); cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice);

    quantize_kernel<<<(n + 255)/256, 256>>>(d_x, d_s, d_z, d_out, n, is_signed, use_float_math);
    
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    FILE *fout = fopen(argv[6], "wb"); fwrite(h_out, 1, bytes, fout); fclose(fout);
    
    free(h_x); free(h_s); free(h_z); free(h_out);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_z); cudaFree(d_out);
    return 0;
}
