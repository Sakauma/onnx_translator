/**
  ******************************************************************************
  * @file        verify_rotary_embedding.cu
  * @author      Egor Izmaylov
  * @brief       提供 RotaryEmbedding 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct RotaryParams {
    int32_t rank;
    int32_t batch_size;
    int32_t num_heads;
    int32_t sequence_length;
    int32_t head_size;
    int32_t rotary_dim;
    int32_t interleaved;
    int32_t has_position_ids;
    int32_t cos_rank;
};

// 根据 ONNX RotaryEmbedding 的原始输入布局生成扁平索引。
__device__ size_t rotary_index(int rank, int b, int h, int s, int d, int num_heads, int sequence_length, int head_size) {
    if (rank == 4) {
        return (((size_t)b * (size_t)num_heads + (size_t)h) * (size_t)sequence_length + (size_t)s) * (size_t)head_size + (size_t)d;
    }
    return ((size_t)b * (size_t)sequence_length + (size_t)s) * ((size_t)num_heads * (size_t)head_size)
         + (size_t)h * (size_t)head_size + (size_t)d;
}

// 实现 RotaryEmbedding CUDA reference kernel，覆盖 position_ids 和 interleaved 两种主要分支。
__global__ void rotary_embedding_kernel(
    const float* x,
    const float* cos_cache,
    const float* sin_cache,
    const long long* position_ids,
    float* out,
    RotaryParams p,
    size_t total
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    int d = 0;
    int s = 0;
    int h = 0;
    int b = 0;
    int hidden = p.num_heads * p.head_size;
    if (p.rank == 4) {
        d = (int)(tid % (size_t)p.head_size);
        s = (int)((tid / (size_t)p.head_size) % (size_t)p.sequence_length);
        h = (int)((tid / ((size_t)p.head_size * (size_t)p.sequence_length)) % (size_t)p.num_heads);
        b = (int)(tid / ((size_t)p.head_size * (size_t)p.sequence_length * (size_t)p.num_heads));
    } else {
        int flat = (int)(tid % (size_t)hidden);
        d = flat % p.head_size;
        h = flat / p.head_size;
        s = (int)((tid / (size_t)hidden) % (size_t)p.sequence_length);
        b = (int)(tid / ((size_t)hidden * (size_t)p.sequence_length));
    }

    if (d >= p.rotary_dim) {
        out[tid] = x[tid];
        return;
    }

    int half = p.rotary_dim / 2;
    int pair = p.interleaved ? d / 2 : (d < half ? d : d - half);
    size_t cache_index = 0;
    if (p.has_position_ids) {
        long long pos = position_ids[(size_t)b * (size_t)p.sequence_length + (size_t)s];
        cache_index = (size_t)pos * (size_t)half + (size_t)pair;
    } else {
        cache_index = ((size_t)b * (size_t)p.sequence_length + (size_t)s) * (size_t)half + (size_t)pair;
    }

    int real_dim = p.interleaved ? pair * 2 : pair;
    int imag_dim = p.interleaved ? pair * 2 + 1 : pair + half;
    size_t x1_index = rotary_index(p.rank, b, h, s, real_dim, p.num_heads, p.sequence_length, p.head_size);
    size_t x2_index = rotary_index(p.rank, b, h, s, imag_dim, p.num_heads, p.sequence_length, p.head_size);
    float x1 = x[x1_index];
    float x2 = x[x2_index];
    float c = cos_cache[cache_index];
    float sn = sin_cache[cache_index];
    out[tid] = (p.interleaved ? (d % 2 == 0) : (d < half)) ? c * x1 - sn * x2 : sn * x1 + c * x2;
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
    // <out_len> <x.bin> <cos.bin> <sin.bin> <position_ids.bin|null> <params.bin> <out.bin>
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <out_len> <x.bin> <cos.bin> <sin.bin> <position_ids.bin|null> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    const char* x_path = argv[2];
    const char* cos_path = argv[3];
    const char* sin_path = argv[4];
    const char* pos_path = argv[5];
    const char* params_path = argv[6];
    const char* out_path = argv[7];

    RotaryParams params;
    FILE* fp = fopen(params_path, "rb");
    if (!fp) { fprintf(stderr, "open params failed\n"); return 1; }
    if (fread(&params, sizeof(RotaryParams), 1, fp) != 1) {
        fprintf(stderr, "read params failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if ((params.rank != 3 && params.rank != 4) || params.rotary_dim <= 0 || (params.rotary_dim % 2) != 0) {
        fprintf(stderr, "invalid params\n");
        return 1;
    }
    size_t expected_len = (params.rank == 4)
        ? (size_t)params.batch_size * (size_t)params.num_heads * (size_t)params.sequence_length * (size_t)params.head_size
        : (size_t)params.batch_size * (size_t)params.sequence_length * (size_t)params.num_heads * (size_t)params.head_size;
    if (out_len != expected_len) {
        fprintf(stderr, "out_len mismatch\n");
        return 1;
    }
    size_t half = (size_t)params.rotary_dim / 2;
    size_t cos_len = params.has_position_ids
        ? 0
        : (size_t)params.batch_size * (size_t)params.sequence_length * half;

    if (params.has_position_ids) {
        FILE* fc = fopen(cos_path, "rb");
        if (!fc) return 1;
        fseek(fc, 0, SEEK_END);
        long bytes = ftell(fc);
        fclose(fc);
        if (bytes <= 0 || bytes % (long)sizeof(float) != 0) return 1;
        cos_len = (size_t)bytes / sizeof(float);
    }

    std::vector<float> h_x(out_len);
    std::vector<float> h_cos(cos_len);
    std::vector<float> h_sin(cos_len);
    std::vector<float> h_out(out_len);
    std::vector<long long> h_pos(params.has_position_ids ? (size_t)params.batch_size * (size_t)params.sequence_length : 0);
    if (!read_vector(x_path, h_x) || !read_vector(cos_path, h_cos) || !read_vector(sin_path, h_sin)) {
        fprintf(stderr, "read input failed\n");
        return 1;
    }
    if (params.has_position_ids && !read_vector(pos_path, h_pos)) {
        fprintf(stderr, "read position ids failed\n");
        return 1;
    }

    float *d_x = NULL, *d_cos = NULL, *d_sin = NULL, *d_out = NULL;
    long long* d_pos = NULL;
    cudaMalloc((void**)&d_x, out_len * sizeof(float));
    cudaMalloc((void**)&d_cos, cos_len * sizeof(float));
    cudaMalloc((void**)&d_sin, cos_len * sizeof(float));
    cudaMalloc((void**)&d_out, out_len * sizeof(float));
    if (params.has_position_ids) {
        cudaMalloc((void**)&d_pos, h_pos.size() * sizeof(long long));
    }
    cudaMemcpy(d_x, h_x.data(), out_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos.data(), cos_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin.data(), cos_len * sizeof(float), cudaMemcpyHostToDevice);
    if (params.has_position_ids) {
        cudaMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(long long), cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (int)((out_len + (size_t)threads - 1) / (size_t)threads);
    rotary_embedding_kernel<<<blocks, threads>>>(d_x, d_cos, d_sin, d_pos, d_out, params, out_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

    fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed\n");
        return 1;
    }
    if (fwrite(h_out.data(), sizeof(float), out_len, fp) != out_len) {
        fprintf(stderr, "write output failed\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    cudaFree(d_x);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_out);
    if (d_pos) cudaFree(d_pos);
    return 0;
}
