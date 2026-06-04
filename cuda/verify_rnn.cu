/**
  ******************************************************************************
  * @file        verify_rnn.cu
  * @author      Egor Izmaylov
  * @brief       提供 RNN 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
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

// 按 layout 读取 X[t,b,i]。
__device__ size_t rnn_x_index(int layout, int seq_len, int batch, int input_size, int t, int b, int i) {
    return layout == 1 ? ((size_t)b * seq_len + t) * input_size + i : ((size_t)t * batch + b) * input_size + i;
}

// 按 layout 写入 Y[t,d,b,h]。
__device__ size_t rnn_y_index(int layout, int seq_len, int num_dirs, int batch, int hidden, int t, int d, int b, int h) {
    return layout == 1
        ? (((size_t)b * seq_len + t) * num_dirs + d) * hidden + h
        : (((size_t)t * num_dirs + d) * batch + b) * hidden + h;
}

// 实现 RNN 主输出 `Y` 的 CUDA 参考 kernel，默认激活为 Tanh。
__global__ void rnn_kernel(
    const double* X,
    const double* W,
    const double* R,
    const double* B,
    const int64_t* sequence_lens,
    const double* initial_h,
    double* Y,
    double* workspace,
    int seq_len,
    int batch,
    int input_size,
    int num_dirs,
    int hidden,
    int direction,
    int layout
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double* h_state = workspace;
    double* h_new = h_state + (size_t)num_dirs * batch * hidden;

    for (int d = 0; d < num_dirs; ++d) {
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < hidden; ++h) {
                size_t idx = ((size_t)d * batch + b) * hidden + h;
                h_state[idx] = initial_h[idx];
            }
        }
    }

    for (int d = 0; d < num_dirs; ++d) {
        int reverse = direction == 1 || (direction == 2 && d == 1);
        for (int step = 0; step < seq_len; ++step) {
            int t = reverse ? (seq_len - 1 - step) : step;
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < hidden; ++h) {
                    double pre = 0.0;
                    for (int i = 0; i < input_size; ++i) {
                        pre += X[rnn_x_index(layout, seq_len, batch, input_size, t, b, i)]
                             * W[((size_t)d * hidden + h) * input_size + i];
                    }
                    for (int hh = 0; hh < hidden; ++hh) {
                        pre += h_state[((size_t)d * batch + b) * hidden + hh]
                             * R[((size_t)d * hidden + h) * hidden + hh];
                    }
                    pre += B[(size_t)d * 2 * hidden + h];
                    pre += B[(size_t)d * 2 * hidden + hidden + h];
                    h_new[(size_t)b * hidden + h] = tanh(pre);
                }
            }

            for (int b = 0; b < batch; ++b) {
                int active = t < (int)sequence_lens[b];
                for (int h = 0; h < hidden; ++h) {
                    size_t state_idx = ((size_t)d * batch + b) * hidden + h;
                    if (active) h_state[state_idx] = h_new[(size_t)b * hidden + h];
                    Y[rnn_y_index(layout, seq_len, num_dirs, batch, hidden, t, d, b, h)] = h_state[state_idx];
                }
            }
        }
    }
}

// 作为 CUDA 验证程序入口，从二进制文件读取输入、执行参考计算并写回结果。
int main(int argc, char** argv) {
    // <out_len> <X> <W> <R> <B> <sequence_lens> <initial_h> <params> <out>
    if (argc != 10) return 1;

    size_t out_len = (size_t)atoll(argv[1]);
    int32_t p[8];
    FILE* fp = fopen(argv[8], "rb");
    if (!fp) return 2;
    if (fread(p, sizeof(int32_t), 8, fp) != 8) {
        fclose(fp);
        return 3;
    }
    fclose(fp);

    int seq_len = p[0], batch = p[1], input_size = p[2], num_dirs = p[3], hidden = p[4], direction = p[5], layout = p[6];
    if (out_len != (size_t)seq_len * num_dirs * batch * hidden) return 4;

    size_t x_len = (size_t)seq_len * batch * input_size;
    size_t w_len = (size_t)num_dirs * hidden * input_size;
    size_t r_len = (size_t)num_dirs * hidden * hidden;
    size_t b_len = (size_t)num_dirs * 2 * hidden;
    size_t h_len = (size_t)num_dirs * batch * hidden;

    double* h_x = (double*)malloc(x_len * sizeof(double));
    double* h_w = (double*)malloc(w_len * sizeof(double));
    double* h_r = (double*)malloc(r_len * sizeof(double));
    double* h_b = (double*)malloc(b_len * sizeof(double));
    int64_t* h_seq = (int64_t*)malloc((size_t)batch * sizeof(int64_t));
    double* h_init = (double*)malloc(h_len * sizeof(double));
    double* h_y = (double*)malloc(out_len * sizeof(double));
    if (!h_x || !h_w || !h_r || !h_b || !h_seq || !h_init || !h_y) return 5;

    FILE* f = fopen(argv[2], "rb"); fread(h_x, sizeof(double), x_len, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(h_w, sizeof(double), w_len, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(h_r, sizeof(double), r_len, f); fclose(f);
    f = fopen(argv[5], "rb"); fread(h_b, sizeof(double), b_len, f); fclose(f);
    f = fopen(argv[6], "rb"); fread(h_seq, sizeof(int64_t), (size_t)batch, f); fclose(f);
    f = fopen(argv[7], "rb"); fread(h_init, sizeof(double), h_len, f); fclose(f);

    double *d_x, *d_w, *d_r, *d_b, *d_init, *d_y, *d_workspace;
    int64_t* d_seq;
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_w, w_len * sizeof(double));
    cudaMalloc(&d_r, r_len * sizeof(double));
    cudaMalloc(&d_b, b_len * sizeof(double));
    cudaMalloc(&d_seq, (size_t)batch * sizeof(int64_t));
    cudaMalloc(&d_init, h_len * sizeof(double));
    cudaMalloc(&d_y, out_len * sizeof(double));
    cudaMalloc(&d_workspace, (h_len + (size_t)batch * hidden) * sizeof(double));
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, w_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, r_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq, h_seq, (size_t)batch * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init, h_init, h_len * sizeof(double), cudaMemcpyHostToDevice);

    rnn_kernel<<<1, 1>>>(d_x, d_w, d_r, d_b, d_seq, d_init, d_y, d_workspace, seq_len, batch, input_size, num_dirs, hidden, direction, layout);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, out_len * sizeof(double), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(argv[9], "wb");
    if (!fo) return 6;
    fwrite(h_y, sizeof(double), out_len, fo);
    fclose(fo);

    free(h_x); free(h_w); free(h_r); free(h_b); free(h_seq); free(h_init); free(h_y);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_r); cudaFree(d_b); cudaFree(d_seq); cudaFree(d_init); cudaFree(d_y); cudaFree(d_workspace);
    return 0;
}
