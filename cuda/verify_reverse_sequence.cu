/**
  ******************************************************************************
  * @file        verify_reverse_sequence.cu
  * @author      Egor Izmaylov
  * @brief       提供 ReverseSequence 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RANK 8

struct ReverseSequenceParams {
    int32_t rank;
    int32_t time_axis;
    int32_t batch_axis;
    int32_t dims[MAX_RANK];
};

__global__ void reverse_sequence_kernel(const float* input, const int64_t* sequence_lens, float* output, ReverseSequenceParams p, size_t out_len) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= out_len) return;

    size_t tmp = tid;
    int32_t coords[MAX_RANK] = {0};
    for (int d = p.rank - 1; d >= 0; --d) {
        int32_t dim = p.dims[d];
        coords[d] = dim > 0 ? (int32_t)(tmp % (size_t)dim) : 0;
        if (dim > 0) {
            tmp /= (size_t)dim;
        }
    }

    int32_t b = coords[p.batch_axis];
    int32_t t = coords[p.time_axis];
    int64_t len = sequence_lens[b];
    if (t < len) {
        coords[p.time_axis] = (int32_t)len - 1 - t;
    }

    size_t src = 0;
    size_t stride = 1;
    for (int d = p.rank - 1; d >= 0; --d) {
        src += (size_t)coords[d] * stride;
        stride *= (size_t)p.dims[d];
    }
    output[tid] = input[src];
}

static int read_params(const char* path, ReverseSequenceParams* p) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open params failed\n");
        return 0;
    }
    int32_t header[3] = {0, 0, 1};
    if (fread(header, sizeof(int32_t), 3, fp) != 3) {
        fclose(fp);
        fprintf(stderr, "read header failed\n");
        return 0;
    }
    p->rank = header[0];
    p->time_axis = header[1];
    p->batch_axis = header[2];
    if (p->rank <= 0 || p->rank > MAX_RANK) {
        fclose(fp);
        fprintf(stderr, "invalid rank\n");
        return 0;
    }
    if (fread(p->dims, sizeof(int32_t), (size_t)p->rank, fp) != (size_t)p->rank) {
        fclose(fp);
        fprintf(stderr, "read dims failed\n");
        return 0;
    }
    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    // <out_len> <input.bin> <sequence_lens.bin> <params.bin> <out.bin>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <out_len> <input.bin> <sequence_lens.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* input_path = argv[2];
    const char* seq_path = argv[3];
    const char* params_path = argv[4];
    const char* out_path = argv[5];

    ReverseSequenceParams params;
    if (!read_params(params_path, &params)) {
        return 1;
    }

    int32_t batch = params.dims[params.batch_axis];
    float* h_input = (float*)malloc(out_len * sizeof(float));
    float* h_output = (float*)malloc(out_len * sizeof(float));
    int64_t* h_seq = (int64_t*)malloc((size_t)batch * sizeof(int64_t));
    if (!h_input || !h_output || !h_seq) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    FILE* fi = fopen(input_path, "rb");
    if (!fi || fread(h_input, sizeof(float), out_len, fi) != out_len) {
        if (fi) fclose(fi);
        fprintf(stderr, "read input failed\n");
        return 1;
    }
    fclose(fi);

    FILE* fs = fopen(seq_path, "rb");
    if (!fs || fread(h_seq, sizeof(int64_t), (size_t)batch, fs) != (size_t)batch) {
        if (fs) fclose(fs);
        fprintf(stderr, "read sequence_lens failed\n");
        return 1;
    }
    fclose(fs);

    float* d_input = NULL;
    float* d_output = NULL;
    int64_t* d_seq = NULL;
    cudaMalloc(&d_input, out_len * sizeof(float));
    cudaMalloc(&d_output, out_len * sizeof(float));
    cudaMalloc(&d_seq, (size_t)batch * sizeof(int64_t));
    cudaMemcpy(d_input, h_input, out_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq, h_seq, (size_t)batch * sizeof(int64_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    reverse_sequence_kernel<<<blocks, threads>>>(d_input, d_seq, d_output, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fo = fopen(out_path, "wb");
    if (!fo || fwrite(h_output, sizeof(float), out_len, fo) != out_len) {
        if (fo) fclose(fo);
        fprintf(stderr, "write output failed\n");
        return 1;
    }
    fclose(fo);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_seq);
    free(h_input);
    free(h_output);
    free(h_seq);
    return 0;
}
