/**
  ******************************************************************************
  * @file        verify_affine_grid.cu
  * @author      Egor Izmaylov
  * @brief       提供 AffineGrid 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

struct AffineGridParams {
    int32_t spatial_rank;
    int32_t N;
    int32_t D;
    int32_t H;
    int32_t W;
    int32_t align_corners;
};

// 将空间下标转换为 AffineGrid 使用的 [-1, 1] 规范化坐标。
__device__ float affine_grid_coord(int index, int size, int align_corners) {
    if (size <= 1) {
        return 0.0f;
    }
    if (align_corners) {
        return -1.0f + 2.0f * (float)index / (float)(size - 1);
    }
    return -1.0f + (2.0f * (float)index + 1.0f) / (float)size;
}

// 实现 2D AffineGrid CUDA 参考 kernel，输出布局为 [N, H, W, 2]。
__global__ void affine_grid_2d_kernel(const float* theta, float* out, AffineGridParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int coord = (int)(tid % 2);
    int w = (int)((tid / 2) % (size_t)p.W);
    int h = (int)((tid / ((size_t)2 * p.W)) % (size_t)p.H);
    int n = (int)(tid / ((size_t)2 * p.W * p.H));

    float x = affine_grid_coord(w, p.W, p.align_corners);
    float y = affine_grid_coord(h, p.H, p.align_corners);
    size_t theta_base = ((size_t)n * 2 * 3) + (size_t)coord * 3;
    out[tid] = theta[theta_base] * x + theta[theta_base + 1] * y + theta[theta_base + 2];
}

// 实现 3D AffineGrid CUDA 参考 kernel，输出布局为 [N, D, H, W, 3]。
__global__ void affine_grid_3d_kernel(const float* theta, float* out, AffineGridParams p, size_t total) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int coord = (int)(tid % 3);
    int w = (int)((tid / 3) % (size_t)p.W);
    int h = (int)((tid / ((size_t)3 * p.W)) % (size_t)p.H);
    int d = (int)((tid / ((size_t)3 * p.W * p.H)) % (size_t)p.D);
    int n = (int)(tid / ((size_t)3 * p.W * p.H * p.D));

    float x = affine_grid_coord(w, p.W, p.align_corners);
    float y = affine_grid_coord(h, p.H, p.align_corners);
    float z = affine_grid_coord(d, p.D, p.align_corners);
    size_t theta_base = ((size_t)n * 3 * 4) + (size_t)coord * 4;
    out[tid] = theta[theta_base] * x + theta[theta_base + 1] * y + theta[theta_base + 2] * z + theta[theta_base + 3];
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <theta.bin> <size.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <theta.bin> <size.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* theta_path = argv[2];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    AffineGridParams p;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 1;
    }
    if (fread(&p, sizeof(AffineGridParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (p.spatial_rank != 2 && p.spatial_rank != 3) {
        fprintf(stderr, "invalid spatial rank\n");
        return 1;
    }
    size_t theta_len = (p.spatial_rank == 2)
        ? (size_t)p.N * 2 * 3
        : (size_t)p.N * 3 * 4;

    std::vector<float> h_theta(theta_len);
    std::vector<float> h_out(out_len);

    fp = fopen(theta_path, "rb");
    if (!fp) {
        fprintf(stderr, "open theta failed\n");
        return 1;
    }
    if (fread(h_theta.data(), sizeof(float), theta_len, fp) != theta_len) {
        fprintf(stderr, "read theta failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    float* d_theta = NULL;
    float* d_out = NULL;
    cudaMalloc((void**)&d_theta, theta_len * sizeof(float));
    cudaMalloc((void**)&d_out, out_len * sizeof(float));
    cudaMemcpy(d_theta, h_theta.data(), theta_len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + threads - 1) / threads);
    if (p.spatial_rank == 2) {
        affine_grid_2d_kernel<<<blocks, threads>>>(d_theta, d_out, p, out_len);
    } else {
        affine_grid_3d_kernel<<<blocks, threads>>>(d_theta, d_out, p, out_len);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open out failed\n");
        cudaFree(d_theta);
        cudaFree(d_out);
        return 1;
    }
    if (fwrite(h_out.data(), sizeof(float), out_len, fp) != out_len) {
        fprintf(stderr, "write out failed\n");
        fclose(fp);
        cudaFree(d_theta);
        cudaFree(d_out);
        return 1;
    }
    fclose(fp);

    cudaFree(d_theta);
    cudaFree(d_out);
    return 0;
}
